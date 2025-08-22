#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Literal, Optional

from pytod.command import CommandCollection
from pytod.iterators import SGDIterator
from pytod.pytod_types.aliases import SPLITS, DialogueID, NodeName, TaskSequence
from pytod.pytod_types.sgd_conversation import Conversation
from pytod.utils import load_json

logger = logging.getLogger(__name__)


def invert_index(
    index_key: list[str], index_path: Path
) -> dict[str, dict[DialogueID, TaskSequence]]:
    """Gives a mapping from dialogue IDs to the flows they represent."""
    node_path = index_path.joinpath(*index_key, "conversations.json")
    node = load_json(node_path)
    inverted_index = {split: {} for split in node}
    for split in node:
        this_split_index: dict[TaskSequence, DialogueID] = node[split]
        for flow, dialogue_ids in this_split_index.items():
            for id in dialogue_ids:
                assert id not in inverted_index[split]
                inverted_index[split][id] = flow
    return inverted_index


def build_nodes(index_key: list[str], index_path: Path, nodes: dict[NodeName, Any]):
    """Add nodes to the path specified by `index_key`."""
    current_leaf = Path(index_path).joinpath(*index_key)
    for node in nodes:
        logger.info(f"Extending path {index_path} with node {node}")
        child = current_leaf.joinpath(node)
        child.mkdir()
        with open(child / "conversations.json", "w") as f:
            json.dump(nodes[node], f)


def get_subtrees(
    *,
    split: Literal["train", "dev", "test"],
    data_path: str,
    index_path: str,
    nodes: Optional[list[list[str]]] = None,
    return_only: Optional[set[str]] = None,
    conversation_builder: Optional[Callable] = None,
    service: Optional[str] = None,
    **kwargs,
) -> Generator[tuple[list[str], Conversation], None, None]:
    """Return the conversation in the subtrees listed in `nodes`. If nodes is
    not specified, then all the conversations in the subtree are returned.
    """

    def leaf_path_to_index_key(index_path: Path, leaf_path: Path) -> list[str]:
        return str(leaf_path.relative_to(index_path)).split("/")

    data_path, index_path = Path(data_path), Path(index_path)
    iterator = SGDIterator(
        data_path, index_path, conversation_builder=conversation_builder
    )
    schema_path = kwargs.get("schema_path")
    if schema_path is not None:
        iterator.command_collections[split] = CommandCollection(schema_path)
    get_leaf_name = partial(leaf_path_to_index_key, index_path)
    if nodes is None:
        nodes = [
            get_leaf_name(leaf_path=pth.parent)
            for pth in index_path.glob("**/*.json")
            if pth.name != "metadata.json"
        ]
        assert isinstance(nodes, list)
        assert all(isinstance(pth, list) for pth in nodes)
        assert all(isinstance(node_, str) for pth in nodes for node_ in pth)
    assert split in SPLITS
    logger.info(f"Iterating through conversation in split: {split}")
    for node in nodes:
        logger.info(f"Visiting node: {node}")
        for _, conversation in iterator.index_iterator(
            split, index_key=node, return_only=return_only, service=service
        ):
            yield list(node), conversation

