#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Generic, Literal, Mapping, Optional, TypeVar

from mongoquery import Query

from pytod.pytod_types.aliases import CanonicalValue, IntentName, ServiceName, SlotName
from pytod.simulation.database_utils import serialise_result

logger = logging.getLogger(__name__)

EntityHash = str
"""A str representation of an SGD entity.
See `serialise_result` for details."""


T = TypeVar("T", bound=Mapping)

RecordInfo = dict[Literal["order", "id"], str]


def remove_attribute(attribute: str, entities: list[T]):
    """In-place removal of `attribute` from every entry in `entities`."""
    for e in entities:
        e.pop(attribute)


def empty_query(parameters: Mapping[str, str]) -> bool:
    """Returns `True` if no query parameters have been specified."""
    return len(parameters) == 1 and "dialogue_id" in parameters or not parameters


class MongoDBCollection(Generic[T]):
    def __init__(self, collection: list[T], split: Literal["dev", "test"]):
        self._collection = collection
        self._split = split


    def query(
        self,
        query: dict[str, str],
        dialogue_id_filtering: Literal["strict", "lenient"] = "strict",
    ) -> list[T]:
        """Query the database.

        Parameters
        ----------
        query
            A mapping from slot names to slot canonical values. The mapping
            also contains the special `dialogue_id` field (see Notes).
        dialogue_id_filtering
            If the query is incorrect, no records retrieved may match the `dialogue_id`.
            If this parameter is set to `strict` then no results are returned. Otherwise,
            the results retrieved if the database if called without the `dialogue_id`
            attribute are returned.

        Notes
        -----

        1. We use the session ID to filter the database results for a number of reasons
            a) For certain services (eg Banks_*/CheckBalance) results are random
            b) Empty queries return random results (eg Music_1/LookupSong)
            c) The same query matches multiple dialogues but some attributes may differ
              (eg Hotels_4/SearchHotel queries return the same records multiple times,
             with differences in attributes such as number_of_rooms/price_per_night)
            d) there is no particular ordering for results (eg Alarm_1/GetAlarms lists alarms
            set but there is no ordering by name or alarm time)


            Using the dialogue ID ensures we return the exact entity the agent mentions in
            conversation and thus that we can do NLG.
        """

        def lowercase_query(query: dict[str, str]) -> dict[str, str]:
            return {k: v.lower() for k, v in query.items()}

        assert dialogue_id_filtering in ["lenient", "strict"], (
            f"Unknown value for dialogue_id_filtering. "
            f"Expected 'lenient' or 'strict', got {dialogue_id_filtering} "
        )
        query = lowercase_query(deepcopy(query))
        if empty_query(query):
            query.update({"matches_empty_query": True})
        else:
            query.update({"matches_empty_query": False})
        q = Query(query)
        raw_matched = [deepcopy(item) for item in self._collection if q.match(item)]
        if not raw_matched and dialogue_id_filtering == "lenient":
            query.pop("dialogue_id")
            q = Query(query)
            raw_matched = [deepcopy(item) for item in self._collection if q.match(item)]
        return self._postprocess_results(raw_matched)

    @staticmethod
    def _postprocess_results(query_results: list[T]) -> list[T]:
        special_keys = ("matches_empty_query", "dialogue_id")
        for key in special_keys:
            remove_attribute(key, query_results)
        return query_results

    def append(self, item: T):
        self._collection.append(item)

    @property
    def num_items(self):
        return len(self._collection)

    @property
    def split(self):
        return self._split


def lowercase_record(
    record: dict[SlotName, CanonicalValue]
) -> dict[SlotName, CanonicalValue]:
    """Lowercase certain attributes of a database record"""
    # the values of these attribs are not
    # lowercased because they are results sorting
    # keys and the results order is different after
    # sorting the lowercased values. As a result,
    # the DB call may no longer ground the results
    # (eg dev/11_00121).
    skip_lowercase = {
        "car_name",
        "attraction_name",
        # "restaurant_name",
    }

    lowercased = {}
    for attrib, value in record.items():
        if attrib in skip_lowercase:
            lowercased[attrib] = value
        else:
            lowercased[attrib] = value.lower()
    return lowercased


def initialise_databases(
    split: Literal["dev", "test"]
) -> Optional[dict[ServiceName, dict[IntentName, MongoDBCollection]]]:
    def init_intent_db(
        entities: list[dict[SlotName, CanonicalValue]],
        entities_to_dial_ids: dict[EntityHash, list[RecordInfo]],
        split: Literal["dev", "test"],
        lowecase: bool = True,
    ) -> MongoDBCollection:
        """Gather entities in a collection than can be queried with a
        MongoDB-like query language.

        Parameters
        ----------
        entities
            A list of unique entities annotating the dialogues.
        entities_to_dial_ids
            The dialogue IDs where an entity appears along with other
            information such as the order of the entity in the annotated
            results list ('order') and the index of the call in the dialogue
            ('call_id').
        split
        lowercase
            Lowercase the attributes of the database entries.
        """

        db_entities = []
        for e in entities:
            e_str = serialise_result(e)
            dial_info: list[RecordInfo] = entities_to_dial_ids[e_str]
            if lowecase:
                e = lowercase_record(e)
            this_e = deepcopy(e)
            for info in dial_info:
                new_e = {**this_e, **info}
                new_e["order"] = int(new_e["order"])
                new_e["call_id"] = int(new_e["call_id"])
                db_entities.append(new_e)
        return MongoDBCollection(db_entities, split)

    assert split in ["dev", "test"], f"No database definition for split {split}"
    if os.getenv("ENTITIES") is not None:
        entities_path = Path(os.getenv("ENTITIES")) / f"{split}.json"
        entities_dial_ids_path = Path(os.getenv("ENTITIES")) / f"{split}_dial_ids.json"
    else:
        logger.error(
            "Could not initialise databases because ENTITIES env var was not set."
            "Have you run setup.sh?"
        )
        return
    with open(entities_path, "r") as f:
        entities = json.load(f)
    with open(entities_dial_ids_path, "r") as f:
        entities_to_ids = json.load(f)

    databases = {service: {} for service in entities}
    for service in databases:
        for intent, this_intent_entities in entities[service].items():
            this_intent_dial_ids = entities_to_ids[service][intent]
            databases[service][intent] = init_intent_db(
                this_intent_entities, this_intent_dial_ids, split
            )
    return databases


dev_databases = initialise_databases("dev")
test_databases = initialise_databases("test")
