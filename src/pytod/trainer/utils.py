#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import re
from pathlib import Path
from typing import Literal, Optional

from omegaconf import ListConfig, OmegaConf
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


_EXPECTED_SCHEMA_VARIANTS = ["v1", "v2", "v3", "v4", "v5"]
HF_SPLITS: list[Literal["train", "validation", "test"]] = [
    "train",
    "validation",
    "test",
]
split = Literal["train", "validation", "test"]
HFTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def infer_schema_variant_from_path(path: str) -> str:
    """Extracts the schema version from the data path."""
    match = re.search(r"\bv[1-9]\b", path)  # noqa
    if match is not None:
        schema_version = path[match.start() : match.end()]
        assert schema_version in _EXPECTED_SCHEMA_VARIANTS
    else:
        schema_version = "original"
    return schema_version


def infer_data_version_from_path(path: str) -> str:
    """Extract the string representing the data version from the data path."""
    # pattern = r"\/(v\d+\.\d+(\.\d+)?|version_(\d+)|(v|\b)(\d+(\.\d+)?))"
    pattern = r"\/(v\d+\.\d+(\.\d+)*|version_(\d+)|v(\d+(\.\d+)*))"
    match = re.search(pattern, path)  # noqa
    if match is not None:
        version = match.group(1) or match.group(3)
    else:
        logger.warning(f"Could not detect data version in path {path}")
        version = ""
    return version


def sgd_variant(test_file_name: str) -> str:
    return infer_schema_variant_from_path(test_file_name)


def resolve_data_version(file_paths: list[str] | str) -> str:
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    model_input_data_versions = list(
        {infer_data_version_from_path(p) for p in file_paths}
    )
    assert (
        len(model_input_data_versions) == 1
    ), "Cannot train on multiple input data versions."
    model_input_data_version = model_input_data_versions[0]
    return model_input_data_version


def make_absolute(data_path_or_paths: str | list[str]) -> str | list[str]:
    if isinstance(data_path_or_paths, ListConfig):
        abs_paths = OmegaConf.create(
            {f"{p}": f"${{root:{p}}}" for p in data_path_or_paths}
        )
    else:
        abs_paths = OmegaConf.create(
            {f"{data_path_or_paths}": f"${{root:{data_path_or_paths}}}"}
        )
    OmegaConf.resolve(abs_paths)
    abs_paths = list(abs_paths.values())
    if isinstance(data_path_or_paths, ListConfig):
        return abs_paths
    return abs_paths[0]


OmegaConf.register_new_resolver(
    "sgd_variant", lambda test_file_name: sgd_variant(test_file_name)
)

OmegaConf.register_new_resolver(
    "resolve_data_version", lambda train_files: resolve_data_version(train_files)
)

OmegaConf.register_new_resolver(
    "make_absolute", lambda train_files: make_absolute(train_files)
)


def infer_checkpoint(
    train_mode: bool,
    output_dir: str,
    resume_from_checkpoint: Optional[str | bool],
    overwrite_output_dir: bool,
) -> Optional[str]:
    """Handle training restarts from checkpoint. The checkpoint path can be specified::

    - by setting`resume_from_checkpoint` to a str representing the checkpoint path
    - by setting `output_dir` to be the checkpoints directory and `resume_from_checkpoint=true`
    """

    def maybe_get_ckpt_path(
        train_mode: bool,
        output_dir: str,
        resume_from_checkpoint: Optional[str | bool],
        overwrite_output_dir: bool,
    ) -> Optional[str]:
        out_dir = Path(output_dir)
        last_checkpoint = None
        if out_dir.is_dir() and train_mode and not overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(out_dir)
            if (
                last_checkpoint is None
                and (n_files := len(list(out_dir.iterdir()))) > 2
            ):
                raise ValueError(
                    f"Output directory ({out_dir}) already exists and is"
                    f" not empty. Use --overwrite_output_dir to overcome. Found {n_files}"
                )
            elif last_checkpoint is not None and resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To train from"
                    " scratch, change `output_dir` or set `trainer_args.overwrite_output_dir=true`"
                )
        return last_checkpoint

    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif (
        last_checkpoint := maybe_get_ckpt_path(
            train_mode, output_dir, resume_from_checkpoint, overwrite_output_dir
        )
    ) is not None:
        checkpoint = last_checkpoint
    return checkpoint