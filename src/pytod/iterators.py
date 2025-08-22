#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel, model_validator

from pytod.command import CommandCollection
from pytod.pytod_types.aliases import DialogueID, ServiceName
from pytod.pytod_types.sgd_conversation import Conversation, Turn
from pytod.sgd_conversation_builder import build_conversation
from pytod.utils import default_to_regular, get_sgd_shard_name, load_json, sort_shards

logger = logging.getLogger(__name__)

SGD_FNAME_PATTERN = re.compile(r"dialogues_[0-9]+\.json")

TASK_SERVICE_SEPARATOR = "|||"
FILTERS_MODULE = "pytod.iterator_utils"


class IndexKeyError(Exception):
    pass


def _validate_leaf_path(leaf_path: Path) -> bool:
    """Checks the path through the index exists"""
    return leaf_path.exists()


class SGDApiNovelty(BaseModel):
    sgd_zero_shot_services: list[str]
    seen_services: list[str]
    zero_shot_services: list[str]


class IndexMetadata(BaseModel):
    services_to_files: dict[
        Literal["train", "dev", "test"], dict[ServiceName, list[str]]
    ]
    split_names: list[Literal["train", "test", "dev"]]
    search_intents: set[str]
    transactional_intents: set[str]
    sgd_api_novelty: Optional[SGDApiNovelty] = None

    @model_validator(mode="before")
    def check_intent_definitions(cls, values: dict):  # type: ignore
        assert not set(values["search_intents"]).intersection(
            values["transactional_intents"]
        )
        return values


def infer_split(pth: Path) -> Literal["train", "dev", "test"]:
    """Given a path, find the innermost subdirectory among `train`, `dev`, and `test`."""

    matches = re.findall(r"/(train|dev|test)/?", str(pth))
    if matches:
        return matches[-1]
    else:
        raise ValueError(
            "No 'train', 'dev', or 'test' subdirectory found in the specified path."
        )


class SGDIterator:
    """Iterate through conversations in SGD."""

    split_names = ["train", "dev", "test"]

    def __init__(
        self,
        data_path: Union[str, Path],
        index_path: Optional[Union[str, Path]] = None,
        conversation_builder: Optional[
            Union[
                Callable[[dict, Any], Union[Conversation, dict[str, Any]]], DictConfig
            ]
        ] = None,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)

        if data_path.exists() and not data_path.is_dir():
            raise ValueError(f"{data_path} already exists but is not a directory")
        self._data_path = data_path
        self._all_files = [r for r in data_path.iterdir()]
        split_paths = self._split_paths()
        if not split_paths:
            split_paths[infer_split(data_path)] = data_path
        self.split_paths = split_paths
        self.schema_paths = self._schema_paths()
        if not self._schema_paths():
            logger.warning(
                "Iterator does not have access to schema, some features may not work."
            )
        self._index_path = None
        self._metadata = None
        if index_path is not None:
            self._index_path = Path(index_path)
            self._metadata = IndexMetadata.model_validate(
                load_json(self._index_path / "metadata.json")
            )
        if conversation_builder is None:
            self._conversation_builder = build_conversation
        else:
            if isinstance(conversation_builder, DictConfig):
                self._conversation_builder = instantiate(conversation_builder)
            else:
                self._conversation_builder = conversation_builder
        self.command_collections: dict[str, CommandCollection] = {}

    def _split_paths(self):
        paths = {}
        for split in SGDIterator.split_names:
            r = [f for f in self._all_files if f.name == split]
            if not r:
                continue
            [paths[split]] = r
        return paths

    def _schema_paths(self):
        return {
            split: self.split_paths[split].joinpath("schema.json")
            for split in SGDIterator.split_names
            if split in self.split_paths
        }

    def _get_filepaths(
        self,
        split: Literal["train", "test", "dev"],
    ) -> list[Path]:
        """Returns a list of file paths for all dialogue batches in a given split.

        Parameters
        ----------
        split
            The split whose filepaths should be returned
        """
        fpaths = sorted(
            list(self.split_paths[split].glob("dialogues_*.json")),
            key=lambda fpath: int((fpath.name.split("_")[1]).split(".")[0]),
        )
        if "dialogues_and_metrics.json" in fpaths:
            fpaths.remove("dialogues_and_metrics.json")
        return fpaths

    @staticmethod
    def _get_dial_key(shard: list[dict]) -> Literal["id", "dialogue_id"]:
        key = "dialogue_id"
        try:
            _ = shard[0][key]
        except KeyError:
            key = "id"
        return key

    @staticmethod
    def _get_max_index(shard: list[dict], key: Literal["id", "dialogue_id"]) -> int:
        """Retrieve the ID of the last dialogue in the shard."""
        return int(shard[-1][key].split("_")[-1]) + 1

    def _file_iterator(
        self, fpath: Path, return_only: Optional[set[str]] = None
    ) -> tuple[str, Union[Conversation, dict[str, Any]]]:
        """
        Iterator through an SGD .json file.

        Parameters
        ----------
        fpath:
            Absolute path to the file.
        return_only
            A set of dialogues to be returned. Specified by dialogue IDs as
            found in the `dialogue_id` file of the schema.
        """

        if not SGD_FNAME_PATTERN.match(fpath.name):
            return
        split = fpath.parent.name
        if split not in SGDIterator.split_names:
            split = infer_split(fpath)
        with open(fpath, "r") as f:
            shard = json.load(f)
        sort_shards({f"{fpath.name}": shard})
        dial_id_key = self._get_dial_key(shard)
        n_dialogues = len(shard)
        try:
            max_index = self._get_max_index(shard, dial_id_key)
        except IndexError:
            max_index = -100
        missing_dialogues = not (max_index == n_dialogues)
        if return_only:
            return_only = sorted(
                list(return_only), key=lambda x: tuple(int(e) for e in x.split("_"))
            )
            if not missing_dialogues:
                for dial_idx in (int(dial_id.split("_")[1]) for dial_id in return_only):
                    yield fpath, self._conversation_builder(
                        dialogue=shard[dial_idx],
                        command_collection=self.command_collections.get(split),
                    )
            else:
                returned = set()
                for dial in shard:
                    found_id = dial[dial_id_key]
                    found_id = found_id.replace(f"{split}_", "")
                    if found_id in return_only:
                        returned.add(found_id)
                        yield fpath, self._conversation_builder(
                            dialogue=dial,
                            command_collection=self.command_collections.get(split),
                        )
                        if returned == set(return_only):
                            break
                if returned != set(return_only):
                    logger.warning(
                        f"Could not find dialogues: {return_only - returned} in shard {fpath.name}"
                    )
        else:
            for dial in shard:
                yield fpath, self._conversation_builder(
                    dialogue=dial,
                    command_collection=self.command_collections.get(split),
                )

    def split_iterator(
        self,
        split: Literal["train", "dev", "test"],
        return_only: Optional[set[str]] = None,
    ) -> tuple[Path, Union[Conversation, dict[str, Any]]]:
        """

        Parameters
        ----------
        split
            Split through which to iterate.
        return_only
            Return only certain dialogues, specified by their schema ``dialogue_id`` field.
        """

        assert (
            split in SGDIterator.split_names
        ), f"Unknown split {split}. Should be one of {SGDIterator.split_names}"
        # return specified dialogues only
        if return_only:
            fpath_map = self._get_file_map(list(return_only), split)
            shards = sorted(
                list(fpath_map.keys()),
                key=lambda x: int(x.name.split(".")[0].split("_")[1]),
            )
            for s in shards:
                yield from self._file_iterator(s, return_only=set(fpath_map[s]))
        # iterate through all dialogues
        else:
            for fp in self._get_filepaths(split):
                yield from self._file_iterator(fp)

    def _get_file_map(
        self,
        dialogue_ids: list[DialogueID],
        split: Literal["train", "test", "dev"],
    ) -> dict[Path, set[DialogueID]]:
        """Returns a map where the keys are file paths and values are lists
        comprising dialogues from `dialogue_ids` that are in the same file.

        dialogue_ids:
            IDs of the dialogues whose paths are to be returned, formatted
            as the schema 'dialogue_id' field.
        split:
            The name of the split whose paths are to be returned.
        """

        file_map = defaultdict(set)
        for dial_id in dialogue_ids:
            # ValueError occurs if dialogue IDs do not match SGD convention
            try:
                fpath = self.split_paths[split].joinpath(get_sgd_shard_name(dial_id))
            except ValueError:
                found_dialogue = False
            else:
                # for the original SGD data, one can reconstruct the filename
                # from dial ID to load the dialogue
                file_map[fpath].add(dial_id)
                continue
            # in general, just iterate through the file to find a given
            # dialogue
            if not found_dialogue:
                for fpath in self.split_paths[split].iterdir():
                    if not fpath.name.startswith("dialogues"):
                        continue
                    with open(fpath, "r") as f:
                        dial_bunch = json.load(f)
                    for dial in dial_bunch:
                        if dial["dialogue_id"] == dial_id:
                            found_dialogue = True
                            break

                    if found_dialogue:
                        break

                if found_dialogue:
                    file_map[fpath].add(dial_id)
                else:
                    logger.warning(f"Could not find dialogue {dial_id}...")

        return default_to_regular(file_map)

    def _service_iterator(
        self,
        service_name: str,
        multi_domain: bool = False,
        target_splits: Optional[list[str]] = None,
        return_only: Optional[set[str]] = None,
    ) -> tuple[str, Union[Conversation, dict[str, Any]]]:
        assert (
            self._metadata is not None
        ), "Please make sure you specify index_path to iter constructor"
        services_to_rfpaths = self._metadata.services_to_files
        if not target_splits:
            splits = [
                split
                for split in self._metadata.split_names
                if service_name in services_to_rfpaths[split]
            ]
        else:
            splits = target_splits
        if not splits:
            raise ValueError(
                f"Service {service_name} was not found in metadata of any split. Is it a typo? "
            )
        root = self._data_path
        for split in splits:
            rel_fpaths = services_to_rfpaths[split][service_name]
            fpaths = {root.joinpath(rfpath): None for rfpath in rel_fpaths}
            if return_only is not None:
                fpaths = self._get_file_map(list(return_only), split)
            for fpath, to_return in fpaths.items():
                for path, dialogue in self._file_iterator(fpath, return_only=to_return):
                    services = dialogue.services
                    if len(services) > 1 and not multi_domain:
                        continue
                    if service_name in services:
                        yield path, dialogue

    def index_iterator(
        self,
        split: Literal["train", "dev", "test"],
        index_key: Optional[list[str]] = None,
        service: Optional[str] = None,
        non_contiguous_service_access: bool = False,
        **kwargs,
    ) -> tuple[Path, Conversation]:
        """Iterates through the SGD index.

        The index is a tree where the nodes are characterise conversations and leaves
        that contain `.json` files summarising conversations of that type. Iteration
        through it enables us to view/summarise subsets of different complexity.

        Parameters
        ----------
        split
        index_key
            This should be a valid list specifying the path to a leaf.
        service
            If specified, only conversations where the APIs of `service` are called
            should be displayed.
        non_contiguous_service_access
            If specified, the iteration is through dialogues where the system access APIs
            from the same service but in distant part of the conversation. For example, the flow

            ..``"Buses_3(FindBus)|||Travel_1(FindAttractions)|||Buses_3(BuyBusTicket)|||Hotels_4(SearchHotel)"``

            meets this criterion because the intent switched to a different service after searching
            for a bus before getting the bus tickets. These dialogues are interesting because the
            user may decline the intent to buy a bus ticket after search and because some of the
            slots needed for the `BuyBusTicket` API call are specified at the beginning during
            the search. To complete the transaction, the user may specify additional, ticket
            related details.

            If `service` is specified, the non-contiguity condition is defined wrt the specified
            service.
        kwargs
            Searched for:
             - `return_only`, which is passed to `split_iterator` if `index_iterator=None`.
             - 'contains_task`, which is used to return only conversation which start with
                a given task

        Examples
        --------
        >>> self.index_iterator('test', ['multi_domain'])
        yields multi-domain conversations from the test corpus
        >>> self.index_iterator('test', ['multi_domain'], non_contiguous_service_access=True)
        yields test set multi-domain conversations where the user changes domain and
        then chakges back to the same domain
        >>> self.index_iterator('dev', ['multi_domain'], non_contiguous_service_access=True, service='Events_1')  # noqa
        yields dev set conversations where the user does something else between
        searching and buying a ticket
        >>> self.index_iterator('test', ['multi_domain', 'search'])
        yields test set multi-domain conversations where the user only searches something
        >>> self.index_iterator('test', ['single_domain'], service='Hotels_3')
        yields convresations where the user searches and books a hotel using the `Hotels_3` API.
        >>> self.index_iterator('test', ['single_intent', 'transactional'], service='Alarm_1')
        yields conversations where the user sets an alarm using the `Alarm_1` API.
        >>> self.index_iterator('test', ['single_intent', 'search', 'changed_goal'])
        yields conversations where the user changes constraints during search
        >>> self.index_iterator('test', ['single_intent', 'search', 'req_alts'])
        yields conversations where the user requests more results without chaning constraints
        >>> self.index_iterator('test')
        yields conversations from the test split

        Notes
        -----
        `return_only` kwarg is not implemented when `index_key` is specified.
        """

        def update_filters(
            filters: list,
            service: Optional[str],
            non_contiguous_service_access: bool,
            **kwargs,
        ):
            """Use kwargs to import filters for displaying only relevant conversations."""
            if service is not None:
                filters.append(
                    hydra.utils.get_object(f"{FILTERS_MODULE}.matches_service")
                )
            if non_contiguous_service_access:
                filters.append(
                    hydra.utils.get_object(
                        f"{FILTERS_MODULE}.has_non_contiguous_service_access"
                    )
                )
            if kwargs.get("contains_task") is not None:
                filters.append(
                    hydra.utils.get_object(f"{FILTERS_MODULE}.contains_task")
                )
            if kwargs.get("startswith_task") is not None:
                filters.append(
                    hydra.utils.get_object(f"{FILTERS_MODULE}.startswith_task")
                )
            return

        # handle basic cases for iteration when no dialogues subset is specified
        if index_key is None:
            if service is None:
                if non_contiguous_service_access:
                    raise ValueError("""Please specify "index_key=['multi_domain']"!""")
                yield from self.split_iterator(
                    split, return_only=kwargs.get("return_only", None)
                )
            else:
                yield from self._service_iterator(
                    service,
                    multi_domain=True,
                    target_splits=[split],
                    return_only=kwargs.get("return_only", None),
                )
            return
        filters = []
        update_filters(filters, service, non_contiguous_service_access)
        leaf_path = self._index_path.joinpath(*index_key, "conversations.json")
        if not _validate_leaf_path(leaf_path):
            raise IndexKeyError(f"Key mismatch for {index_key}")
        try:
            conversations = load_json(leaf_path)[split]
        except KeyError:
            conversations = load_json(leaf_path)
            logger.warning(
                f"Key {index_key} was not found in split {split}\n"
                f"Dialogues matching this key are in {list(conversations.keys())}"
            )
            return
        for flow, dial_ids in conversations.items():
            to_return = kwargs.get("return_only", None)
            if to_return is not None:
                assert to_return, "Set ids to None, not empty list!"
                dial_ids = set(to_return).intersection(dial_ids)
                if not dial_ids:
                    continue
            if all(filt(flow, service, **kwargs) for filt in filters):
                yield from self.split_iterator(split, return_only=dial_ids)


def turn_pair_iterator(conversation: Conversation) -> tuple[Turn, Turn]:
    """Iterate through (user, system) turn pairs."""
    iterator = iter(conversation.turns)
    for first in iterator:
        second = next(iterator, None)
        yield first, second
