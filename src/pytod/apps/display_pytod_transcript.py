#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from importlib import resources
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pytod.iterators import SGDIterator

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / ".")


def get_datapath(cfg: DictConfig) -> Path | str:
    if (experiment := cfg.experiment_name) is None:
        return cfg.data_path
    variant, ckpt, ver = cfg.sgd_variant, cfg.checkpoint, cfg.version
    suffix = f"{experiment}/{variant}/{cfg.split}/{ver}/checkpoint-{ckpt}"
    return Path(cfg.hyps_dir) / suffix


@hydra.main(
    config_name="display_pytod_transcript.yaml",
    config_path=get_config_path(),
)
def display_pytod_transcript(cfg: DictConfig):
    iterator = SGDIterator(
        get_datapath(cfg),
        conversation_builder=instantiate(cfg.conversation_builder),
    )
    formatter = instantiate(cfg.formatter)
    display = (
        getattr(formatter, "rich_display")
        if cfg.use_rich
        else getattr(formatter, "conversation_to_text")
    )
    print()
    for _, dialogue in iterator.split_iterator(
        cfg.split, return_only=set(cfg.ids) if cfg.ids is not None else None
    ):
        print(f"Dialogue({dialogue.id}:{cfg.split})")
        print(display(dialogue))
        print()


if __name__ == "__main__":
    display_pytod_transcript()
