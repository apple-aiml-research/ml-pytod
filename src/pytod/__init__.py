#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from dotenv import load_dotenv

# environment should always be loaded first
# because database init requires reading the
# .env file
load_dotenv()

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover  # noqa
from pathlib import Path  # noqa

from omegaconf import DictConfig, OmegaConf  # noqa

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


def _resolve_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _parent(path: str) -> Path:
    return Path(path).parent


def _subfield(node: DictConfig, field: str):
    return node[field]


def apply_patch_version(input_data_version: str, patch_version: int | None) -> str:
    """Applies a patch version to the input data version."""
    if patch_version is None:
        return input_data_version
    return f"{input_data_version}.{patch_version}"


OmegaConf.register_new_resolver(
    "apply_patch_version",
    lambda input_data_version, patch_version: apply_patch_version(
        input_data_version, patch_version
    ),
)
OmegaConf.register_new_resolver("subfield", _subfield)
OmegaConf.register_new_resolver("root", lambda path: f"{_resolve_root() / path}")
OmegaConf.register_new_resolver("parent", lambda path: f"{_parent(path)}")
