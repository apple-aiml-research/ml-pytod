#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from mongoquery import Query
from omegaconf import DictConfig, OmegaConf

from pytod.iterators import SGDIterator
from pytod.preprocessing_pipelines import modify_slot_names
from pytod.pytod_types.aliases import DialogueID, ServiceName, SlotName, SlotValue
from pytod.sgd_utils import dialogue_iterator
from pytod.utils import load_json, load_resources, rename_key, save_json, write_shards

logger = logging.getLogger(__name__)

SGD_SPLITS = ["train", "dev", "test"]


TurnIdx = int
Result = dict[str, str]
# a database record / api response


def correct_query_result_mismatches(dial: dict[str, Any]):
    """In the RentalCars_1, RentalCars_3 services, the value
    'pickup_time' in the query does not match the
    results' annotation.

    We correct annotations as follows:

        1. We discard records that do not match the query
        2. If no results match the query then we replace original
        mismatched slot with the value in the query parameters.
    """

    # TODO: GENERALISE AND ALSO APPLY TO SGD-X
    def get_mismatched_results(results: list[dict[str, str]], query: Query):
        """Return the results which do not match the query"""
        return [r for r in results if not query.match(r)]

    def replace_mismatched_values(
        results: list[dict[str, str]], parameters: dict[str, str]
    ):
        """Replace mismatched values with query parameters."""
        for r in results:
            for slot, value in r.items():
                if slot in parameters and value != (query_value := parameters[slot]):
                    logger.info(
                        f"Replacing slot {slot} value {value} with {query_value}"
                    )
                    r[slot] = query_value

    if not any(
        service not in dial["services"] for service in ["RentalCars_1", "RentalCars_3"]
    ):
        return
    for idx, turn in enumerate(dialogue_iterator(dial)):
        for frame_ in turn["frames"]:
            if "service_call" in frame_:
                intent = frame_["service_call"]["method"]
                if intent != "GetCarsAvailable":
                    continue
                parameters = frame_["service_call"]["parameters"]
                results = frame_["service_results"]
                query = Query(parameters)
                mismatched_results = get_mismatched_results(results, query)
                if mismatched_results:
                    logging.info(
                        f"{dial['dialogue_id']}:turn {idx}. Correcting database annotations"
                    )
                    n_mismatched = len(mismatched_results)
                    if n_mismatched == len(results):
                        logger.info(
                            f"{dial['dialogue_id']}:turn {idx}. No records matched the query"
                        )
                        replace_mismatched_values(results, parameters)
                    else:
                        n_orig_results = len(results)
                        logger.info(
                            f"Dropping {n_mismatched} out of {n_orig_results} annotated records"
                        )
                        frame_["service_results"] = [
                            r for r in results if r not in mismatched_results
                        ]
                        assert (
                            len(frame_["service_results"])
                            == n_orig_results - n_mismatched
                        )


def correct_canonical_value_annotations(
    dial: dict[str, Any], corrections: dict[ServiceName, dict[SlotName, dict]]
):
    """The database call annotations use canonical value annotations but
    for a limited number of slots these are not consistent (eg "sushi" is
    mapped to "Sushi" and "Japanese" in different dialogues). To simplify
    normalisation_orig, we automatically re-annotate the corpus such that there
    is one canonical form for each slot value.
    """

    AnnotatedValue = str
    CorrectedValue = str
    remap_params_only = {"restaurant_name"}

    def correct_action_annotations(
        annotation_map: defaultdict,
    ) -> dict[SlotName, dict[AnnotatedValue, CorrectedValue]]:
        for a in frame_["actions"]:
            match a["act"]:
                case "OFFER" | "CONFIRM" | "INFORM":
                    if (slot := a["slot"]) in corrections[service]:
                        value = a["values"][0]
                        if value in corrections[service][slot]:
                            expected_canonical_value = corrections[service][slot][value]
                            canonical_value = a["canonical_values"][0]
                            if expected_canonical_value != canonical_value:
                                a["canonical_values"] = [expected_canonical_value]
                                logger.info(
                                    f"Dialogue: {dial['dialogue_id']}. Service: {service}. "
                                    f"Slot: {slot}. Remapped canonical annotation "
                                    f"{canonical_value} to {expected_canonical_value}"
                                )
                                annotation_map[slot][
                                    canonical_value
                                ] = expected_canonical_value
        return dict(annotation_map)

    def correct_service_calls_and_results(
        reannotated_values: dict[SlotName, dict[AnnotatedValue, CorrectedValue]]
    ):
        if "service_call" in frame_:
            updates = {}
            for param, value in (
                parameters := frame_["service_call"]["parameters"]
            ).items():
                if param in reannotated_values:
                    if value in reannotated_values[param]:
                        updates[param] = reannotated_values[param][value]
                        logger.info(
                            f"Dialogue: {dial['dialogue_id']}. Service: {service}. Slot: {param}. "
                            f"Remapped parameter value {value} to {updates[param]}"
                        )
                        if param not in remap_params_only:
                            logger.info("Remapping results")
                            for r in (results := frame_["service_results"]):
                                r[param] = updates[param]
                            logger.info(f"Remapped {len(results)} results")
            parameters.update(updates)

    if not set(dial["services"]).intersection(corrections.keys()):
        return

    reannotated_values = defaultdict(dict)
    for idx, turn in enumerate(dialogue_iterator(dial)):
        for frame_ in turn["frames"]:
            service = frame_["service"]
            if service not in corrections:
                continue
            correct_action_annotations(reannotated_values)
            correct_service_calls_and_results(reannotated_values)


def add_missing_results_annotations(
    dial: dict[str, Any],
    missing_results: dict[ServiceName, dict[DialogueID, dict[str, list[Result]]]],
):
    """Result annotations grounding the conversation are missing from a few of dev set dialogues."""

    if set(dial["services"]).intersection(missing_results.keys()):
        dialogue_id = dial["dialogue_id"]
        for idx, turn in enumerate(dialogue_iterator(dial)):
            idx = str(idx)
            for frame in turn["frames"]:
                service = frame["service"]
                if (
                    service in missing_results
                    and dialogue_id in missing_results[service]
                    and idx in missing_results[service][dialogue_id]
                ):
                    logger.info(
                        f"Adding missing results annotations to {dialogue_id} (dev)"
                    )
                    new_results = missing_results[service][dialogue_id][idx]
                    frame["service_results"].extend(new_results)


def apply_utterance_corrections(
    dial: dict[str, Any], corrections_lookup: dict[str, str]
):
    """Apply corrections to utterances that are incorrect with respect to the original
    annotation."""
    if (dial_id := dial["dialogue_id"]) in corrections_lookup:
        logger.info(f"Applying utterance corrections to {dial_id}")
        corrections = corrections_lookup[dial_id]
        for turn in dialogue_iterator(dial):
            if (original := turn["utterance"]) in corrections:
                correction = corrections[turn["utterance"]]
                logger.info(f"Correcting '{original}' to '{correction}'")
                turn["utterance"] = correction


def correct_default_values(dev_schema: list[dict]):
    """Correct the default value for the `flight_class`
    slot in Flights_3 service"""

    logger.info("Correcting Flights_3 default values in dev schema")
    for service_schema in dev_schema:
        if service_schema["service_name"] == "Flights_3":
            for intent_schema in service_schema["intents"]:
                for slot in intent_schema["optional_slots"]:
                    if slot == "flight_class":
                        intent_schema["optional_slots"][slot] = "dontcare"
                        break


def remap_python_keyword_slot_names(
    dial: dict[str, Any], slot_name_map: dict[ServiceName, dict[SlotName, str]]
):
    """Two slot names in the Trains_1 service are reserved python keywords.
    We remap them to new names to ensure function calls are AST-parseable."""

    def remap_slot_names(
        turn: dict[str, Any],
        service: ServiceName,
        slot_map: dict[str, str],
    ):
        """In-place remapping of argument names of certain SGD APIs
        to ensure function calls are AST-parseable."""

        def modify_actions(frame: dict[str, Any]):
            actions = frame["actions"]
            for action in actions:
                if (sgd_slot := action["slot"]) in slot_map:
                    action["slot"] = slot_map[sgd_slot]

        def modify_states(frame: dict[str, Any]):
            state = frame["state"]["slot_values"]
            for to_replace_slot in slot_map:
                rename_key(state, to_replace_slot, slot_map[to_replace_slot])
            requested = frame["state"]["requested_slots"]
            in_reqs = set(requested).intersection(slot_map.keys())
            for to_replace_slot in in_reqs:
                requested.pop(to_replace_slot)
                requested.append(slot_map[to_replace_slot])

        def modify_database_annotations(frame: dict[str, Any]):
            if "service_call" in frame:
                for illegal_slot_name in slot_map:
                    rename_key(
                        frame["service_call"]["parameters"],
                        illegal_slot_name,
                        slot_map[illegal_slot_name],
                    )
                    for result in frame["service_results"]:
                        rename_key(
                            result, illegal_slot_name, slot_map[illegal_slot_name]
                        )

        def modify_span_annotations(frame: dict[str, Any]):
            spans = frame["slots"]
            for span_dict in spans:
                if (slot := span_dict["slot"]) in slot_map:
                    span_dict["slot"] = slot_map[slot]

        for frame in turn["frames"]:
            if frame["service"] != service:
                continue
            modify_actions(frame)
            modify_span_annotations(frame)
            if turn["speaker"] == "USER":
                modify_states(frame)
            if turn["speaker"] == "SYSTEM":
                modify_database_annotations(frame)

    if set(dial["services"]).intersection(slot_name_map.keys()):
        logger.info(
            f"{dial['dialogue_id']}: "
            "Assigning new names to slots which are python reserved keywords"
        )
        for turn in dialogue_iterator(dial):
            for service in slot_name_map:
                remap_slot_names(turn, service, slot_name_map[service])


def remap_python_keyword_slot_names_in_schema(
    schema: list[dict[str, Any]], slot_map: dict[ServiceName, dict[SlotName, str]]
) -> list[dict[str, Any]]:
    """Two slot names in the Trains_1 service are reserved python keywords.
    We remap them to new names to ensure function calls are AST-parseable."""
    logger.info("Remapping python reserved keywords from test schema to valid values")
    for service in slot_map:
        schema = modify_slot_names(schema, service, slot_map[service])
    return schema


def remove_intent_from_schema(schema: list[dict[str, Any]]):
    """Remove the Movies_1 BuyMoviesTicket from the schema because it does
    not appear in conversation."""

    for service_schema in schema:
        if service_schema["service_name"] == "Movies_1":
            service_schema["intents"] = [
                intent_schema
                for intent_schema in service_schema["intents"]
                if intent_schema["name"] != "BuyMovieTickets"
            ]


def update_categorical_values(
    schema: list[dict[str, Any]],
    updated_values: dict[ServiceName, dict[SlotName, list[SlotValue]]],
):
    """Update the possible values of categorical slots to remove values that
    never appear in the data."""
    for service_schema in schema:
        if (service := service_schema["service_name"]) in updated_values:
            this_service_slot_updates = updated_values[service]
            for slot_info in service_schema["slots"]:
                if (slot_name := slot_info["name"]) in this_service_slot_updates:
                    slot_info["possible_values"] = this_service_slot_updates[slot_name]
                    logger.info(
                        f"Updated possible values for slot {slot_name} in service {service}"
                    )


def correct_schema_definitions():
    pass
    # TODO: DESTINATION/ORIGIN_ARIPORT_NAME HAVE WRONG DESCRIPTION
    #  IN FLIGHTS_3


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


@hydra.main(config_name="annotation_correction.yaml", config_path=get_config_path())
def correct_annotations(config: DictConfig):
    """Apply correction annotations to SGD dialogues."""
    logger.info(OmegaConf.to_yaml(config, resolve=True))
    split = config.split
    out_dir = Path(config.out_dir) / split
    assert split in SGD_SPLITS, f"Unknown split: {split}"
    corrected_dialogues = defaultdict(list)
    resources = load_resources(config.resource_paths)
    canonical_value_corrections = resources["canonical_value_corrections"]
    all_utterance_corrections = resources["utterance_corrections"]
    python_keyword_slot_map = resources["python_keywords"]
    iterator = SGDIterator(
        config.data_path,
        conversation_builder=instantiate(config.conversation_builder),
    )
    for fpath, dial in iterator.split_iterator(
        config.split,
        return_only=set(config.ids) if config.ids is not None else None,
    ):
        fname = fpath.name
        correct_query_result_mismatches(dial)
        correct_canonical_value_annotations(dial, canonical_value_corrections)
        corrected_dialogues[fname].append(dial)
        if split == "dev":
            apply_utterance_corrections(dial, all_utterance_corrections["dev"])
            add_missing_results_annotations(dial, resources["missing_results"]["dev"])
        if split == "test":
            apply_utterance_corrections(dial, all_utterance_corrections["test"])
            add_missing_results_annotations(dial, resources["missing_results"]["test"])
            remap_python_keyword_slot_names(dial, python_keyword_slot_map)
        if split == "train":
            apply_utterance_corrections(dial, all_utterance_corrections["train"])

    write_shards(corrected_dialogues, out_dir)
    schema = load_json(fpath.parent / "schema.json")
    if split == "train":
        remove_intent_from_schema(schema)
        categorical_value_updates = resources["schema_corrections"][
            "categorical_value_updates"
        ]
        categorical_value_updates = categorical_value_updates["train"]
        update_categorical_values(schema, categorical_value_updates)
    if split == "dev":
        correct_default_values(schema)
    if split == "test":
        schema = remap_python_keyword_slot_names_in_schema(
            schema, python_keyword_slot_map
        )
    save_json(schema, out_dir / "schema.json")


if __name__ == "__main__":
    correct_annotations()
