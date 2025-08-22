#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
from copy import deepcopy
from functools import partial
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pytod.command import CommandCollection
from pytod.iterators import SGDIterator
from pytod.pytod_conversation_builder import pytod_transcript_builder
from pytod.sgd_conversation_builder import no_op_conversation_builder


@pytest.fixture
def command_collection(request):
    split = request.param["split"][0]
    version = request.param["version"]
    if isinstance(version, list):
        version = version[0]
    cfg = OmegaConf.create(
        {"path": f"${{root:data/processed/sgd/{version}/{split}/schema.json}}"}
    )
    return CommandCollection(schema_path=cfg.path)


@pytest.fixture
def train_command_collection(request):
    version = request.param["version"]
    cfg = OmegaConf.create(
        {"path": f"${{root:data/processed/sgd/{version}/train/schema.json}}"}
    )
    return CommandCollection(schema_path=cfg.path)


@pytest.fixture
def restaurant_query(command_collection, request):
    from pytod.simulation.services.restaurants_2 import FindRestaurants

    entities = request.param
    restaurant_query = FindRestaurants.build(
        "", command_collection.get("Restaurants_2", "FindRestaurants")
    )
    restaurant_query._raw_entities = entities
    return restaurant_query, deepcopy(entities), restaurant_query.entity_name


@pytest.fixture
def sgd_iterator(request):
    split = request.param
    cfg = OmegaConf.create({"path": "${root:data/interim/sgd}"})
    iterator = SGDIterator(
        cfg.path,
        conversation_builder=no_op_conversation_builder,
    )
    return iterator.split_iterator(split), split


@pytest.fixture
def test_resource_root() -> Path:
    cfg = OmegaConf.create({"path": f"${{root:tests/resources/dst_testing}}"})  # noqa
    return Path(cfg.path)


@pytest.fixture
def pytod_iterator(request, test_resource_root):
    split = request.param["split"][0]
    version = request.param["version"]
    iterator = SGDIterator(
        test_resource_root / version / split,
        conversation_builder=partial(pytod_transcript_builder),
    )
    return iterator.split_iterator(split), split
