#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from importlib import resources
from typing import Literal, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import CanonicalValue, SlotName, SlotValue
from pytod.sgd_utils import dialogue_iterator
from pytod.utils import nested_defaultdict, save_json

logger = logging.getLogger(__name__)

SGD_SPLITS = ["train", "dev", "test"]


def get_transaction_status(frame_: dict) -> Literal["SUCCESS", "FAILURE"]:
    """Returns a flag indicating whether the transaction executed in the
    current frame succeeds or not."""
    if any(a["act"] == "NOTIFY_SUCCESS" for a in frame_["actions"]):
        return "SUCCESS"
    return "FAILURE"


def get_alternatives(
    parameters: dict[SlotName, CanonicalValue], frame_: dict
) -> Optional[dict[SlotName, list[SlotValue]]]:
    """Return a proposed parameter change by the agent following an API call."""

    actions = {}
    for a in frame_["actions"]:
        if a["act"] == "OFFER":
            actions[a["slot"]] = a["canonical_values"]
    if actions:
        alternatives = {}
        for p, p_val in parameters.items():
            if p in actions and p_val not in actions[p]:
                alternatives[p] = actions[p]
        return alternatives
    return


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


@hydra.main(config_name="api_simulation_build.yaml", config_path=get_config_path())
def gather_transaction_responses(config: DictConfig):
    split = config.split
    logger.info(f"Extracting API responses for split: {split}")
    assert split in SGD_SPLITS, f"Unknown split: {split}"
    call_responses = nested_defaultdict(list, depth=2)
    cmd_collection = instantiate(config.command_collection)
    iterator = SGDIterator(
        config.data_path,
        conversation_builder=instantiate(config.conversation_builder),
    )
    for fpath, dial in iterator.split_iterator(
        config.split,
        return_only=set(config.ids) if config.ids is not None else None,
    ):
        for idx, turn in enumerate(dialogue_iterator(dial)):
            for frame_ in turn["frames"]:
                if "service_call" in frame_:
                    intent = frame_["service_call"]["method"]
                    service = frame_["service"]
                    if not cmd_collection.get(service, intent).is_transactional:
                        continue
                    parameters = frame_["service_call"]["parameters"]
                    status = get_transaction_status(frame_)
                    alternative = None
                    match status:
                        case "FAILURE":
                            alternative = get_alternatives(parameters, frame_)
                    parameters.update(
                        {
                            "status": status,
                            "alternative": alternative,
                            "dialogue_id": dial["dialogue_id"],
                        }
                    )
                    # copy entity properties to the response, as these may be carried
                    # over to other intents
                    if frame_["service_results"]:
                        assert len(frame_["service_results"]) == 1
                        for key in (entity_info := frame_["service_results"][0]):
                            if key not in parameters:
                                parameters[key] = entity_info[key]
                    call_responses[frame_["service"]][intent].append(parameters)
    save_json(call_responses, f"{split}.json")


if __name__ == "__main__":
    gather_transaction_responses()
