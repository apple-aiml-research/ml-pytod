#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Extend the slot value annotations of specific slots with normalised values."""
import logging
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import CanonicalValue, ServiceName, SlotName, SlotValue
from pytod.sgd_utils import dialogue_iterator
from pytod.utils import load_resources, write_shards

logger = logging.getLogger(__name__)

SGD_SPLITS = ["train", "dev", "test"]

NormalisationLookup = dict[ServiceName, dict[SlotName, dict[SlotValue, CanonicalValue]]]

dial_id: str = ""
turn_idx: int = -1


def add_normalised_values(
    dial: dict[str, Any],
    normalisation_lookup: NormalisationLookup,
    in_scope_slots: Optional[dict[ServiceName, list[SlotName]]] = None,
):
    """Extend the value lists of in-scope slots with normalised values.
    If `in_scope_slots` not specified, all value annotations are
    extended."""

    def extend_value_list(
        slot: SlotName,
        values: list[SlotValue],
        slot_canonical_values: dict[SlotValue, CanonicalValue],
        canonical_map: NormalisationLookup,
    ):
        additional_values = []
        for value in values:
            if value in slot_canonical_values:
                additional_values.append(slot_canonical_values[value])
        if not additional_values:
            logger.warning(
                f"Dialogue: {dial_id}. "
                f"Could not find normalised form for slot `{slot}`. "
                f"Values: {values}"
            )
            for c_service, slot_canonical_vals in canonical_map.items():
                for c_slot, canonical_value_dict in slot_canonical_vals.items():
                    for value in values:
                        if (
                            value in canonical_value_dict
                            and canonical_value_dict[value] not in additional_values
                        ):
                            logger.info(
                                f"Dialogue: {dial_id}. "
                                f"Extended with normalised form of `{c_slot}` from `{c_slot}`"
                            )
                            additional_values.append(canonical_value_dict[value])
        for value in additional_values:
            if value not in values:
                logger.info(
                    f"Dialogue: {dial_id}, turn {turn_idx}. "
                    f"Adding value {value} for slot {slot}."
                )
                values.append(value)

    if not set(dial["services"]).intersection(normalisation_lookup.keys()):
        return
    update_scope = in_scope_slots is None
    in_scope_slots = in_scope_slots or {}
    for idx, turn in enumerate(dialogue_iterator(dial)):
        global turn_idx
        turn_idx = idx
        if turn["speaker"] == "SYSTEM":
            continue
        for frame in turn["frames"]:
            states = frame["state"]["slot_values"]
            if not in_scope_slots:
                in_scope_slots[frame["service"]] = states
            if update_scope:
                in_scope_slots[frame["service"]] = states
            if (service := frame["service"]) not in in_scope_slots:
                continue
            for slot, values in states.items():
                if service not in normalisation_lookup:
                    continue
                if slot in in_scope_slots[service]:
                    extend_value_list(
                        slot,
                        values,
                        normalisation_lookup[service][slot],
                        normalisation_lookup,
                    )


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


@hydra.main(config_name="state_annotation.yaml", config_path=get_config_path())
def extend_state_annotations(config: DictConfig):
    logger.info("Adding normalised values to state annotations")
    split = config.split
    out_dir = Path(config.out_dir) / split
    assert split in SGD_SPLITS, f"Unknown split: {split}"
    annotated_dialogues = defaultdict(list)
    resources = load_resources(config.resource_paths)
    normalisation_lookup = resources["normalisation_lookup"]
    in_scope_slots = resources["in_scope_slots"]
    iterator = SGDIterator(
        config.data_path,
        conversation_builder=instantiate(config.conversation_builder),
    )
    for fpath, dial in iterator.split_iterator(
        config.split,
        return_only=set(config.ids) if config.ids is not None else None,
    ):
        global dial_id
        dial_id = dial["dialogue_id"]
        add_normalised_values(dial, normalisation_lookup, in_scope_slots)
        annotated_dialogues[fpath.name].append(dial)
    write_shards(dict(annotated_dialogues), out_dir)


if __name__ == "__main__":
    extend_state_annotations()
