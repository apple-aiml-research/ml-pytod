#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.command import CommandCollection, ServiceCommand, is_arg
from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import IntentName, ServiceName, SlotName
from pytod.sgd_utils import dialogue_iterator, get_user_informed_slots
from pytod.utils import write_shards

logger = logging.getLogger(__name__)

SGD_SPLITS = ["train", "dev", "test"]
METADATA_KEYS = Literal["task_completed", "followup_task"]
WILDCARD_VALUE = "dontcare"
UserTurn = dict[str, Any]
UserFrame = dict[str, Any]


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


def metadata_factory() -> dict[METADATA_KEYS, bool]:
    return {"api_call_required": False, "followup_task": False, "call_parameters": []}


def get_intent_start_frame(
    intent: IntentName, service: ServiceName, user_turns: list[UserTurn]
) -> UserFrame:
    """Backtrack to the first frame where `intent` was expressed by
    the user."""
    prev_frame = None
    for turn in reversed(user_turns):
        try:
            [frame_] = [f for f in turn["frames"] if f["service"] == service]
        except ValueError:
            break
        if frame_["state"]["active_intent"] != intent:
            return prev_frame
        prev_frame = frame_
    assert prev_frame is not None
    return prev_frame


def is_followup(
    intent: IntentName, service: ServiceName, prev_user_turns: list[UserTurn]
) -> bool:
    """Determine if a task is a "follow up" task, that is, whether some
    of its arguments have been specified during previous tasks and should
    be carried over."""
    start_frame = get_intent_start_frame(intent, service, prev_user_turns)
    assert start_frame["service"] == service, (
        start_frame["state"]["active_intent"] == intent
    )
    informed_slot = get_user_informed_slots(start_frame) or set()
    carried_over = {
        s for s in start_frame["state"]["slot_values"] if s not in informed_slot
    }
    return bool(carried_over)


def should_skip_frame(
    system_frame: dict[str, Any], prev_user_frame: dict[str, Any]
) -> bool:
    """We do not evaluate task completion on frames where the user
    requests more results and there are none to show. In the annotation,
    the system frames are annotated with a call, but we do not evaluate
    these frames because the state of the dialogue does not change."""

    notified_failure = any(
        a["act"] == "NOTIFY_FAILURE" for a in system_frame["actions"]
    )
    new_constraints = get_user_informed_slots(prev_user_frame)
    user_confirmed = any(a["act"] == "AFFIRM" for a in prev_user_frame["actions"])
    if notified_failure and not user_confirmed:
        return not bool(new_constraints)
    return False


def get_call_params(
    system_frame: dict[str, Any],
    prev_user_frame: dict[str, Any],
    command_schema: ServiceCommand,
) -> list[SlotName]:
    """We only require that the slots relevant for the current task
    are jointly accurate to evaluate system task completion. These
    are annotated in the system frames. We also add parameters for
    which the user specified a wildcard value if they are part of
    the current API (these are not annotated)."""
    params = list(system_frame["service_call"]["parameters"].keys())
    for slot, values in prev_user_frame["state"]["slot_values"].items():
        if WILDCARD_VALUE in values and is_arg(slot, command_schema):
            assert slot not in params
            params.append(slot)
    return params


@hydra.main(
    config_name="system_task_completion_annotation.yaml", config_path=get_config_path()
)
def annotate_task_completion(config: DictConfig):
    split = config.split
    logger.info(OmegaConf.to_yaml(config, resolve=True))
    logger.info(f"Annotating task completion for split: {split}")
    assert split in SGD_SPLITS, f"Unknown split: {split}"
    schema: CommandCollection = instantiate(config.command_collection)
    annotated_dialogues = defaultdict(list)
    iterator = SGDIterator(
        config.data_path,
        conversation_builder=instantiate(config.conversation_builder),
    )
    for fpath, dial in iterator.split_iterator(
        config.split,
        return_only=set(config.ids) if config.ids is not None else None,
    ):
        fname = fpath.name
        annotated_dialogues[fname].append(dial)
        for idx, turn in enumerate(dialogue_iterator(dial)):
            for frame_ in turn["frames"]:
                if idx > 0:
                    prev_frames = dial["turns"][idx - 1]["frames"]
                else:
                    prev_frames = []
                if "service_call" in frame_ and not should_skip_frame(
                    frame_, prev_frames[0]
                ):
                    intent = frame_["service_call"]["method"]
                    service = frame_["service"]
                    [prev_user_frame] = [
                        f for f in prev_frames if f["service"] == service
                    ]
                    prev_user_turns = [
                        t for t in dial["turns"][:idx] if t["speaker"] == "USER"
                    ]
                    prev_user_frame["metadata"] = {
                        "api_call_required": True,
                        "followup_task": is_followup(intent, service, prev_user_turns),
                        "call_parameters": get_call_params(
                            frame_, prev_user_frame, schema.get(service, intent)
                        ),
                    }
                    for f in prev_frames:
                        if f["service"] != service:
                            f["metadata"] = metadata_factory()
                else:
                    for frame in prev_frames:
                        if "state" in frame:
                            frame["metadata"] = metadata_factory()
    logger.info(f"Annotation complete for split {config.split}")
    write_shards(annotated_dialogues, Path(config.out_dir) / split)


if __name__ == "__main__":
    annotate_task_completion()
