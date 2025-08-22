#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Create dialogue templates for storing model predictions in SGD format during evaluation."""
import json
import logging
import os
from importlib import resources

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def create_template(data):
    """Remove state, policy and span annotations from
    user and system frames."""
    for dialogue in data:
        for turn in dialogue["turns"]:
            if turn["speaker"] == "USER":
                for frame in turn["frames"]:
                    frame["actions"] = []
                    frame["slots"] = []
                    frame["state"] = {
                        "active_intent": "",
                        "requested_slots": [],
                        "slot_values": {},
                    }
            else:
                for frame in turn["frames"]:
                    frame["actions"] = []
                    frame["slots"] = []


def get_config_path() -> str:
    return str(resources.files("pytod.configs") / "pytod_setup")


@hydra.main(config_name="create_templates.yaml", config_path=get_config_path())
def main(config: DictConfig):
    logger.info(
        f"Split: {config.split}. "
        f"Creating SGD dialogue files without annotations for storing evaluation results."
    )
    for root, dirs, files in os.walk(config.input_dir):
        for file in files:
            if file.startswith("dialogues") and "metrics" not in file:
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    create_template(data)
                with open(file, "w") as f:
                    json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
