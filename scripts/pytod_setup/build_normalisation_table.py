#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from importlib import resources

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import CanonicalValue, ServiceName, SlotName, SlotValue
from pytod.sgd_utils import dialogue_iterator
from pytod.utils import load_resources, nested_defaultdict, save_json

logger = logging.getLogger(__name__)


def extend_with_carried_over_values(
    canonical_value_map: dict,
    additional_values: dict[
        ServiceName, dict[SlotName, dict[SlotValue, CanonicalValue]]
    ],
):
    """Extend the automatically extracted canonical map with additional
    value pairs to support mapping of values carried over from different services."""

    logger.info(
        "Extending canonical map with additional values from carried over slots"
    )
    for service in additional_values:
        for slot, value_pairs in additional_values[service].items():
            for v in value_pairs:
                if v not in canonical_value_map[service][slot]:
                    canonical_value_map[service][slot][v] = value_pairs[v]


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


@hydra.main(config_name="normalisation.yaml", config_path=get_config_path())
def build_normalisation_table(config: DictConfig):
    """Build a mapping from the surface form of slot values mentioned in conversation
    to a single, normalised value (canonical value) that is recognised by external APIs
    and databases."""
    logger.info(OmegaConf.to_yaml(config, resolve=True))
    splits = config.splits
    logger.info(
        f"Building normalisation lookup tables using data from splits: {config.splits}"
    )
    canonical_value_map = nested_defaultdict(dict, depth=3)
    for split in splits:
        iterator = SGDIterator(
            config.data_path,
            conversation_builder=instantiate(config.conversation_builder),
        )
        for fpath, dial in iterator.split_iterator(
            split,
            return_only=set(config.ids) if config.ids is not None else None,
        ):
            for turn in dialogue_iterator(dial):
                for frame in turn["frames"]:
                    service = frame["service"]
                    for action in frame["actions"]:
                        match action["act"]:
                            case "INFORM" | "CONFIRM" | "OFFER" | "SELECT":
                                if not action["slot"]:
                                    continue
                                this_slot_canonical_values = canonical_value_map[
                                    service
                                ][action["slot"]]
                                for val, canonical_val in zip(
                                    action["values"], action["canonical_values"]
                                ):
                                    if (
                                        val in this_slot_canonical_values
                                        or val.lower() in this_slot_canonical_values
                                    ):
                                        if val in this_slot_canonical_values:
                                            expected_value = this_slot_canonical_values[
                                                val
                                            ]
                                        else:
                                            expected_value = this_slot_canonical_values[
                                                val.lower()
                                            ]
                                        try:
                                            assert expected_value == canonical_val
                                        except AssertionError:
                                            dial_id = dial["dialogue_id"]
                                            logging.warning(
                                                f"{split}:{service}:{dial_id}: "
                                                f"slot '{action['slot']}' had canonical value "
                                                f"'{expected_value}' for value {val} but "
                                                f"attempted to add another canonical val: "
                                                f"{canonical_val}"
                                            )
                                    this_slot_canonical_values[val] = canonical_val
                                    # make lookup case-insensitive on the input side
                                    if val != val.lower():
                                        if (
                                            val.lower()
                                            not in this_slot_canonical_values
                                        ):
                                            this_slot_canonical_values[
                                                val.lower()
                                            ] = canonical_val
    resources = load_resources(config.resource_paths)
    extend_with_carried_over_values(
        canonical_value_map, resources["carryover_slots_normalised_values"]
    )
    save_json(canonical_value_map, "sgd_canonical_value_map.json")


if __name__ == "__main__":
    build_normalisation_table()
