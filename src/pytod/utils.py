#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import random
import re
import shutil
import subprocess
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import repeat
from operator import methodcaller
from pathlib import Path
from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Type,
    Union,
    cast,
)

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from pytod.pytod_types.aliases import DialogueID, ShardName

logger = logging.getLogger(__name__)


def get_datetime() -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn.enabled
    torch.backends.cudnn.enabled = args.cudnn.deterministic
    torch.backends.cudnn.benchmark = args.cudnn.benchmark


def set_seed_no_gpu(args):
    random.seed(args.seed)
    np.random.seed(args.seed)


def default_to_regular(d: defaultdict) -> dict:
    """Recursively converts a defaultdict `d` to a normal dictionary."""
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def load_json(path: Union[str, Path]):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: Any, path: Union[str, Path], indent: int = 4):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def nested_defaultdict(default_factory: Callable, depth: int = 1):
    """Creates a nested default dictionary of arbitrary depth with a specified callable as leaf."""
    if not depth:
        return default_factory()
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


def dispatch_on_value(func: Callable) -> Callable:
    """
    Value-dispatch function decorator.

    Transforms a function into a value-dispatch function,
    which can have different behaviors based on the value of the first argument.
    """

    registry = {}

    def dispatch(value: Hashable):
        try:
            return registry[value]
        except KeyError:
            return func

    def register(value: Hashable, func: Callable = None):
        def add_to_register(func):
            register(value, func)

        if func is None:
            return add_to_register

        registry[value] = func

        return func

    def wrapper(*args: Any, **kw: Any):
        return dispatch(args[0])(*args, **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


def filter_shards(
    ffs: list[Path], return_only: Optional[set[DialogueID]] = None
) -> tuple[list[Path], Optional[dict[ShardName, list[DialogueID]]]]:
    """Helper to iterate only through shards where specified dialogues are contained."""
    if return_only is not None:
        relevant_shards = set()
        shards_to_dials: dict[ShardName, list[DialogueID]] = defaultdict(list)
        for dial_id in return_only:
            shard_name = get_sgd_shard_name(dial_id)
            relevant_shards.add(shard_name)
            shards_to_dials[shard_name].append(dial_id)
        return [f for f in ffs if f.name in relevant_shards], shards_to_dials
    return ffs, None


def write_shards(
    conversations: dict[ShardName, list[dict]], out_dir: Path, indent: int = 4
):
    for shard, this_shard_data in conversations.items():
        logger.info(f"Writing shard: {shard}")
        save_json(this_shard_data, out_dir / shard, indent=indent)


def sort_shards(conversations: dict[ShardName, list[dict]]):
    def sort_by_dial_number(dials: list[dict]):
        """In-place sorting of dialogue lists by dialogue number."""
        try:
            dials.sort(key=lambda x: int(x["dialogue_id"].split("_")[-1]))
        except KeyError:
            # id: {split}_{dialogue_id}
            dials.sort(key=lambda x: int(x["id"].split("_")[-1]))

    for shard, dials in conversations.items():
        sort_by_dial_number(dials)


def copy_schema(schema_path: Path, out_dir: Path):
    logger.info(f"Copied schema from {schema_path} to {out_dir}")
    shutil.copy(schema_path, out_dir)


def split_camel_case(camel_case_string: str) -> tuple[str, str]:
    # Define the regular expression pattern to match CamelCase
    pattern = r"([A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$))"

    # Find all matches of the pattern in the string
    matches = re.findall(pattern, camel_case_string)

    # The first word is the first match, and the remaining words
    # are the rest of the matches joined together
    first_word = matches[0]
    remaining_words = "".join(matches[1:])

    return first_word, remaining_words


def snake_case(camel_str: str) -> str:
    """Convert method name from camel case to PEP8 style.

    Example
    -------
        "FindRestaurant" -> "find_restaurant"
    """

    snake_str = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str)
    snake_str = re.sub(r"(?<=[a-z])(?=\d)", "_", snake_str)
    snake_str = snake_str.lower()
    if not snake_str:
        return camel_str
    return snake_str


def swap_keys_and_values(d: dict) -> dict[Any, Any]:
    random_val = d[random.choice(list(d.keys()))]
    val_type = type(random_val)
    assert all(isinstance(v, val_type) for v in d.values())
    if val_type is str or val_type is int:
        return {v: k for k, v in d.items()}
    elif val_type is list:
        assert all(
            isinstance(el, str) or isinstance(el, int) for v in d.values() for el in v
        )
        new_dict = {}
        for d_k, d_v in d.items():
            for d_v_el in d_v:
                assert d_v_el not in new_dict
                new_dict[d_v_el] = d_k
        return new_dict
    else:
        raise NotImplementedError(
            f"Value type {val_type} not supported for key-val swapping."
        )


def cast_vals_to_sorted_list(d: dict, sort_by: Optional[callable] = None) -> dict:
    """Casts the values of a nested dict to sorted lists.

    Parameters
    ----------
    d
    sort_by:
        A callable to be used as sorting key.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            cast_vals_to_sorted_list(value)
        else:
            d[key] = sorted(list(value), key=sort_by)

    return d


def append_to_values(result: dict, new_data: dict):
    """Recursively appends to the values of `result` the values in
    `new_data` that have the same keys. If the keys in `new_data`
    do not exist in `result`, they are recursively added. The keys of
    `new_data` can be either lists or single float objects that
    are to be appended to existing `result` keys. List concatenation is
    performed in former case.

    Parameters
    ----------
    result
        Mapping whose values are to be extended with corresponding values from
        `new_data_map`
    new_data
        Data with which the values of `result_map` are extended.
    """

    def dict_factory():
        return defaultdict(list)

    for key in new_data:
        # recursively add any new keys to the result mapping
        if key not in result:
            if isinstance(new_data[key], dict):
                result[key] = dict_factory()
                append_to_values(result[key], deepcopy(new_data[key]))
            else:
                if isinstance(new_data[key], float):
                    result[key] = [new_data[key]]
                elif isinstance(new_data[key], list):
                    result[key] = [*new_data[key]]
                elif isinstance(new_data[key], set):
                    result[key] = new_data[key]
                else:
                    raise ValueError("Unexpected key type.")
        # updated existing values with the value present in `new_data_map`
        else:
            if isinstance(result[key], dict):
                append_to_values(result[key], new_data[key])
            else:
                if isinstance(new_data[key], list):
                    result[key] += new_data[key]
                elif isinstance(new_data[key], float):
                    result[key].append(new_data[key])
                elif isinstance(new_data[key], set):
                    result[key] = result[key].union(new_data[key])
                else:
                    raise ValueError(f"Unexpected key type: {type(key)}")


def store_data(data, store: defaultdict, store_fields: list[str]):
    """Stores data in a nested default dictionary at a specified location.

    Parameters
    ----------
    data
        Data to be stored. If this is a list and the store location is a list,
        the store location is extended with the new elements. If data is not a list,
        then it is appended to the store location.
    store
        Nested default dictionary where data is to be stored.
    store_fields
        Key where the data is to be stored

    Notes
    -----
    Supported types for `store` leaves are ``int``, ``list`` and ``dict``.
    """

    def safeget(dct: dict, *keys: Union[tuple[str], list[str]]):
        """Retrieves the value of one nested key represented in `keys`"""
        for key in keys:
            try:
                dct = dct[key]
            except KeyError:
                return None
        return dct

    try:
        store_location = safeget(store, *store_fields)
    except TypeError:
        raise TypeError
    if isinstance(store_location, list):
        if isinstance(data, list):
            store_location.extend(data)
        else:
            store_location.append(data)
    elif isinstance(store_location, int) or isinstance(store_location, float):
        store_location = safeget(store, *store_fields[:-1])
        if isinstance(data, int) or isinstance(data, float):
            if isinstance(store_location, defaultdict):
                if isinstance(store_location[store_fields[-1]], int) or isinstance(
                    store_location[store_fields[-1]], float
                ):
                    store_location[store_fields[-1]] += data
                else:
                    raise NotImplementedError
    elif isinstance(store_location, dict):
        # nb, this can be more generic
        assert isinstance(data, dict)

        assert set(data.keys()).isdisjoint(store_location.keys())
        store_location.update(data)
    else:
        raise NotImplementedError


def drop_empty_value_keys(dict_: dict[str, Any]):
    """In-place removal of values which evaluate to `False`."""

    to_remove = {key for key, value in dict_.items() if not value}
    while to_remove:
        next_key = to_remove.pop()
        assert any(isinstance(next_key, t) for t in (list, set))
        dict_.pop(next_key)


def typed_partial(cls, *args, **kwargs):
    return cast(Type[cls], partial(cls, *args, **kwargs))


def rename_key(mapping_: Mapping, old_name: Any, new_name: Any):
    """In-place rename key `old_name` as `new_name`."""
    if old_name not in mapping_:
        return
    val = mapping_.pop(old_name)
    mapping_[new_name] = val


def load_resources(mapping_: Mapping) -> dict:
    """Recursively read the contents of a nested dictionary `mapping_` where
    the leaves are absolute paths to .json files."""

    def process_value(value):
        """Process a single value in the mapping."""
        if isinstance(value, str) or isinstance(value, Path):
            path = Path(value)
            if path.suffix.lower() == ".json":
                return load_json(path)
            else:
                return value
        elif isinstance(value, Mapping):
            return load_resources(value)
        else:
            return value

    return {key: process_value(value) for key, value in mapping_.items()}


def count_nested_dict_values(d: dict) -> dict:
    """Returns a mapping with the same structure as the
    input but with the values replaced by the counts of the value fields."""

    count_d = deepcopy(d)

    def _helper(d: dict, count_d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                _helper(value, count_d[key])
            else:
                count_d[key] = len(value)

    _helper(d, count_d)

    return count_d


def listify_values(mapping: Mapping[str, Any]) -> dict[str, list]:
    """Wrap mapping values to lists."""
    return {s: [v] for s, v in mapping.items()}


def cast_vals_to_string(mapping: Mapping) -> dict[Any, str]:
    """Cast values to string"""
    return {s: str(v) for s, v in mapping.items()}


def camel_to_snake_case(name: Optional[str]) -> Optional[str]:
    if name is None:
        return
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(word: str) -> str:
    words = word.split("_")
    return "".join(x.capitalize() or "_" for x in words[0:])


def load_resolved_config(config_path: str) -> DictConfig:
    """Load a .yaml config file using OmegaConf, resolving the interpolations"""
    cfg = OmegaConf.load(Path(config_path))
    OmegaConf.resolve(cfg)
    return cfg


def load_hydra_config(config_dir: Path | str) -> DictConfig:
    """Discovers the hydra configuration in the `config` directory and loads it."""
    if isinstance(config_dir, str):
        config_dir = Path(config_dir)
    if config_dir.is_file():
        config_dir = config_dir.parent
    config_pth = config_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(Path(config_pth))
    OmegaConf.resolve(cfg)
    return cfg


def get_commit_hash():
    """Returns the commit hash for the current HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()


def create_dir(pth: Path | str):
    if isinstance(pth, str):
        pth = Path(pth)
    if not pth.exists():
        pth.mkdir(parents=True)

    return str(pth)


OmegaConf.register_new_resolver("create_dir", lambda pth: create_dir(Path(pth)))


def stringify_values(vals: list[str]) -> str:
    if not vals:
        return ""
    vals = ", ".join(vals) if len(vals) > 1 else f"{vals[0]}"
    return f" ({vals} or other)"


def stringify_list(lst: list[str], word: str = "and") -> Optional[str]:
    if lst:
        if len(lst) == 1:
            return f"{lst[0]}"
        if len(lst) == 2:
            return f"{lst[0]} {word} {lst[1]}"
        return f"{', '.join(lst[:-1])} {word} {lst[-1]}"
    logger.warning("Listify called with an empty list?")
    return


def get_sgd_shard_name(dial_id: str) -> str:
    """Reconstruct SGD filename from dialogue ID."""

    file_prefix = int(dial_id.split("_")[0])

    if file_prefix in range(10):
        str_file_prefix = f"00{file_prefix}"
    elif file_prefix in range(10, 100):
        str_file_prefix = f"0{file_prefix}"
    else:
        str_file_prefix = f"{file_prefix}"

    return f"dialogues_{str_file_prefix}.json"


def aggregate_values(
    mapping: dict, agg_fcn: Literal["mean", "prod"], reduce: bool = True
):
    """Aggregates the values of the input (nested) mapping according to the
    specified aggregation method. This function modifies the input in place.

    Parameters
    ---------
    mapping
        The mapping to be aggregated.
    agg_fcn
        Aggregation function. Only  `mean` or `prod` aggregation supported.
    reduce
        If False, the aggregator will keep the first dimension of the value to be
        aggregated.

    Example
    -------
    >>> mapping = {'a': {'b': [[1, 2], [3, 4]]}}
    >>> agg_fcn = 'mean'
    >>> aggregate_values(mapping, agg_fcn, reduce=False)
    >>> {'a': {'b': [1.5, 3.5]}}

    """

    for key, value in mapping.items():
        if isinstance(value, dict):
            aggregate_values(mapping[key], agg_fcn, reduce=reduce)
        else:
            if reduce:
                aggregator = methodcaller(agg_fcn, value)
                mapping[key] = aggregator(np)
            else:
                if isinstance(mapping[key], list) and isinstance(mapping[key][0], list):
                    agg_res = []
                    for val in mapping[key]:
                        aggregator = methodcaller(agg_fcn, val)
                        agg_res.append(aggregator(np))
                    mapping[key] = agg_res
                else:
                    aggregator = methodcaller(agg_fcn, value)
                    mapping[key] = aggregator(np)


def random_subset(items: list, n_items: int | None = None) -> list | None:
    if not items:
        return
    if n_items is None:
        n_items = random.choice(range(1, len(items)))
    return random.sample(items, n_items)


def load_dialog_histories(data_dir: str) -> dict[DialogueID, list[dict]]:
    """Load the user and system turns alongside with relevant service and
    API metadata."""
    ffs = sorted(Path(data_dir).glob("dialogues*.json"))
    all_histories = {}
    for ff in ffs:
        histories = load_json(ff)
        assert not set(all_histories.keys()).intersection(histories.keys())
        all_histories.update(histories)
    return all_histories
