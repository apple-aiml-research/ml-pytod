#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Sample dialogues with diverse structure."""
import json
import logging
from importlib import resources
from pathlib import Path

import hydra
from omegaconf import DictConfig

from pytod.utils import load_json

logger = logging.getLogger(__name__)


SGD_SPLITS = ["train", "dev", "test"]


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


@hydra.main(config_name="sample_dialogues.yaml", config_path=get_config_path())
def sample_from_index(config: DictConfig):
    for split in SGD_SPLITS:
        this_split_dialogues = []
        logger.info(f"Sampling diverse dialogues for {split} set.")
        opath = Path(config.output_dir) / f"{split}"
        if not opath.exists():
            opath.mkdir(parents=True, exist_ok=True)
        for dial_type in config.index_path:
            for path in config.index_path[dial_type]:
                this_path_conversations = load_json(path)
                if split not in this_path_conversations:
                    continue
                for flow in this_path_conversations[split]:
                    this_split_dialogues.append(this_path_conversations[split][flow][0])
        this_split_dialogues = sorted(
            this_split_dialogues,
            key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])),
        )
        with open(opath / "dialogues.json", "w") as f:
            json.dump(this_split_dialogues, f, indent=4)


if __name__ == "__main__":
    sample_from_index()
