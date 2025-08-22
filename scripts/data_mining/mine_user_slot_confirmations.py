#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from collections import defaultdict
from copy import deepcopy
from functools import partial
from importlib import resources
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from pytod.iterators import SGDIterator
from pytod.sgd_conversation_builder import mine_user_slot_confirmations
from pytod.utils import (
    append_to_values,
    cast_vals_to_sorted_list,
    default_to_regular,
    nested_defaultdict,
)

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    return str(resources.files("pytod.configs.pytod.data_mining") / ".")


@hydra.main(
    config_name="mine_user_slot_confirmations.yaml",
    config_path=get_config_path(),
)
def mine_confirmed_slots(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    all_multivalue_slots = defaultdict()
    run_dir = Path(cfg.run_dir)
    for split in cfg.splits:
        split_path = run_dir / split
        if not split_path.exists():
            logger.info(f"Creating output directory {split_path}")
            split_path.mkdir(exist_ok=False, parents=True)
        collector = nested_defaultdict(set, depth=2)
        setattr(SGDIterator, "user_slot_confirmation_collector", collector)
        iterator = SGDIterator(
            cfg.data_path,
            conversation_builder=partial(
                mine_user_slot_confirmations, user_slot_confirmation_collector=collector
            ),
        )
        logger.info(f"Mining split: {split}")
        for _, dialogue in iterator.split_iterator(
            split,
            return_only=set(cfg.ids) if cfg.ids is not None else None,
        ):
            logger.info(f"Mining dialogue({dialogue.id}:{split})")
        this_split_mined_data = cast_vals_to_sorted_list(
            default_to_regular(iterator.user_slot_confirmation_collector)
        )
        append_to_values(all_multivalue_slots, this_split_mined_data)
        this_split_out = split_path / f"{split}_user_confirmed_slots.json"
        with open(this_split_out, "w") as f:
            json.dump(this_split_mined_data, f)
        OmegaConf.save(
            config=OmegaConf.create(default_to_regular(this_split_mined_data)),
            f=split_path / f"{split}_user_confirmed_slots.yaml",
        )
    with open(run_dir / "user_confirmed_slots.json", "w") as f:
        json.dump(all_multivalue_slots, f)
    OmegaConf.save(
        config=OmegaConf.create(default_to_regular(all_multivalue_slots)),
        f=run_dir / "user_confirmed_slots.yaml",
    )


if __name__ == "__main__":
    mine_confirmed_slots()
