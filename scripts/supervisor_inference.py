#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from importlib import resources
from pathlib import Path

import hydra
from datasets import Dataset, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.evaluation.evaluation_utils import infer_step, log_to_wandb
from pytod.inference.policy_supervisor import UNDEFINED, PolicySupervisor
from pytod.trainer.utils import resolve_data_version

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    return str(resources.files("pytod.configs.pytod_finetuning.trainer") / "assistants")


def create_dataset(
    split: str,
    path: str,
    first_turn_split_data_path: str,
    cache_dir: str,
    eval_first_turn: bool = False,
    max_samples: int | None = None,
    extension: str = "parquet",
):
    pth = first_turn_split_data_path if eval_first_turn else path
    data_version = resolve_data_version(pth)
    logger.info(f"Loading {split} dataset, version {data_version}")
    if eval_first_turn:
        logger.info("Evaluating first turn split")
    raw_dataset: Dataset = load_dataset(extension, data_files=pth, cache_dir=cache_dir)[
        "train"
    ]
    if max_samples is not None:
        raw_dataset = raw_dataset.select(range(max_samples))
    return raw_dataset


@hydra.main(config_name="policy_supervisor", config_path=get_config_path())
def supervisor_inference(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    dataset = create_dataset(
        split=cfg.split,
        path=cfg.data_path,
        first_turn_split_data_path=cfg.first_turn_split_data_path,
        eval_first_turn=cfg.eval_first_turn,
        cache_dir=cfg.architecture.config.cache_dir,
        max_samples=cfg.max_samples,
    )
    supervisor: PolicySupervisor = instantiate(cfg.policy_supervisor)
    supervisor.set_dataset(dataset)
    supervisor.predict()
    supervisor.save_predictions_cache(cfg.cache_dir)
    metrics = supervisor.save_metrics(cfg.cache_dir)
    logger.info(f"Supervisor exact match: {supervisor.exact_match:.2f}%")
    seen_em, unseen_em = (
        supervisor.seen_api_exact_match,
        supervisor.unseen_api_exact_match,
    )
    if seen_em != UNDEFINED:
        logger.info(f"Supervisor exact match (seen): {seen_em:.2f}%")
    if unseen_em != UNDEFINED:
        logger.info(f"Supervisor exact match (unseen): {unseen_em:.2f}%")
    log_to_wandb(cfg.wandb, metrics, infer_step(cfg.model_name_or_path))


if __name__ == "__main__":
    supervisor_inference()
