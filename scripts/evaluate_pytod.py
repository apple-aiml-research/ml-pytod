#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Literal

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pytod.evaluation.evaluation_utils import (
    PER_FRAME_OUTPUT_FILENAME,
    EvaluatorInputs,
    SGDMetrics,
    compute_dst_metrics,
    save_evaluator_outputs,
)
from pytod.execution.pytod_agent import AgentArgs
from pytod.execution.session_executor import execute_and_evaluate
from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import DialogueID, ServiceName, StateDict
from pytod.sgd_utils import get_turn_level_services
from pytod.utils import get_sgd_shard_name, write_shards

logger = logging.getLogger(__name__)

TurnIdx = int
PredictionsDict = (
    dict[
        DialogueID,  # eg 1_00000
        dict[
            TurnIdx,
            dict[Literal["state", "utterance"], dict[ServiceName, StateDict]],
        ],
    ],
)


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / ".")


def _get_suffix(cfg):
    variant, ckpt, ver = cfg.sgd_variant, cfg.checkpoint, cfg.version
    suffix = f"{cfg.experiment_name}/{variant}/{cfg.split}/{ver}/checkpoint-{ckpt}"
    return suffix


def get_hyp_path(cfg: DictConfig) -> Path | str:
    """Return the path where the hypotheses are saved"""
    if cfg.experiment_name is None:
        return cfg.data_path
    return Path(cfg.hyps_dir) / _get_suffix(cfg)


def get_metrics_path(cfg: DictConfig) -> Path | str:
    """Return the path where metrics should be saved."""
    return Path(cfg.metrics_dir) / _get_suffix(cfg)


@hydra.main(
    config_name="evaluate_pytod.yaml",
    config_path=get_config_path(),
)
def exec_and_evaluate(cfg: DictConfig):
    hyps_dir = Path(get_hyp_path(cfg))
    iterator = SGDIterator(
        hyps_dir,
        conversation_builder=instantiate(cfg.conversation_builder),
    )
    schema = instantiate(cfg.command_collection)
    predictions = defaultdict(dict)
    all_refs: EvaluatorInputs = instantiate(cfg.reference_files)
    agent_args = AgentArgs(
        schema=schema,
        split=cfg.split,
        lenient=cfg.lenient,
        debug=cfg.debug,
    )
    total = 0
    dialogues_to_map = []
    for shard_path, dialogue in iterator.split_iterator(
        cfg.split,
        return_only=set(cfg.ids) if cfg.ids is not None else None,
    ):
        logger.debug(f"Dialogue({dialogue.id}:{cfg.split})")
        this_dial_id = dialogue.id.replace(f"{cfg.split}_", "")
        dialogues_to_map.append(this_dial_id)
        service_info = get_turn_level_services(all_refs["dataset_ref"][this_dial_id])
        state = execute_and_evaluate(dialogue, agent_args, service_info)
        predictions[get_sgd_shard_name(this_dial_id)].update(state)
        total += 1
    mapper = instantiate(cfg.mapper, dialogues_to_map=dialogues_to_map)
    shards = mapper.convert_to_sgd_format(predictions)
    metrics: SGDMetrics = compute_dst_metrics(all_refs, shards)
    jga = metrics.metrics_for_logging["#ALL_SERVICES/joint_goal_accuracy"]
    logger.info(f"JGA: {jga}")
    save_evaluator_outputs(
        hyps_dir,
        get_metrics_path(cfg),
        {
            f"model_{cfg.checkpoint}_metrics.json": metrics.service_metrics,
            "service_errors.json": metrics.collect_execution_errors(),
            PER_FRAME_OUTPUT_FILENAME: metrics.per_frame_metrics,
        },
        cfg.checkpoint,
    )
    if cfg.save_shards:
        sgd_format = hyps_dir.joinpath("sgd_format")
        sgd_format.mkdir(parents=True, exist_ok=True)
        write_shards(shards, sgd_format)


if __name__ == "__main__":
    exec_and_evaluate()
