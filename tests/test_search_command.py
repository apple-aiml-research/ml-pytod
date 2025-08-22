#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.simulation._command_utils import SearchError
from pytod.simulation.confirmed_command import ConfirmedCommand
from pytod.simulation.entities import get_entity
from pytod.simulation.search_command import SearchCommand
from pytod.utils import listify_values

TEST_ON_SPLITS = ["dev"]
VERSION = "v0.9.1"

data_version_params = {"split": TEST_ON_SPLITS, "version": VERSION}


@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.local
def test_build(command_collection):
    find_restaurants_command = command_collection.get(
        "Restaurants_2", "FindRestaurants"
    )

    find_restaurants = SearchCommand.build("1_00000", find_restaurants_command)
    assert isinstance(find_restaurants, SearchCommand)
    assert not isinstance(find_restaurants, ConfirmedCommand)
    assert find_restaurants.positional_args == ("category", "location")
    assert find_restaurants.keyword_args == {
        "price_range": "dontcare",
        "has_seating_outdoors": "dontcare",
        "has_vegetarian_options": "dontcare",
    }
    assert find_restaurants.entity_cmp_key == ["restaurant_name"]
    assert find_restaurants.system_tracked_slots == ("restaurant_name",)
    assert find_restaurants.entity_name == "Restaurant"
    assert find_restaurants.followup_command == "ReserveRestaurant"
    assert find_restaurants.service == "Restaurants_2"
    assert find_restaurants.entity_attributes is not None
    assert find_restaurants.entity_attributes
    assert not {
        "price_range",
        "has_seating_outdoors",
        "has_vegetarian_options",
    }.difference(set(find_restaurants._enum_args.keys()))


@pytest.mark.parametrize(
    "restaurant_query",
    [
        [{"restaurant_name": "A"}, {"restaurant_name": "B"}],
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.local
def test___select__(restaurant_query, command_collection):
    query, entities, entity_name = restaurant_query
    first_entity = next(query)
    second_entity = next(query)
    selected = query.__select__(entity=second_entity)
    assert query.selected_entity == second_entity
    assert selected == second_entity
    value = "A"
    selected_by_value = query.__select__(key=query.entity_cmp_key[0], value=value)
    assert query.selected_entity == selected_by_value
    assert getattr(selected_by_value, query.entity_cmp_key[0]) == value
    assert selected_by_value == first_entity

    with pytest.raises(AssertionError):
        query.__select__(key=query.current_entity)
    with pytest.raises(SearchError):
        unknown_entity = get_entity(
            entity_name, {"restaurant_name": "C"}, cmp_attributes=query.entity_cmp_key
        )
        query.__select__(entity=unknown_entity)


@pytest.mark.parametrize(
    "restaurant_query",
    [[{"restaurant_name": "A"}, {"restaurant_name": "A"}, {"restaurant_name": "B"}]],
    indirect=True,
)
@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.local
def test___next__(restaurant_query, command_collection):
    def get_unique_entities(entities):
        uniq = []
        for e in entities:
            if e not in uniq:
                uniq.append(e)
        return uniq

    query, entities, entity_name = restaurant_query
    unique_entities = list(reversed(get_unique_entities(entities)))
    while unique_entities:
        expected = get_entity(
            entity_name, unique_entities.pop(), cmp_attributes=query.entity_cmp_key
        )
        actual = next(query)
        assert expected == actual

    with pytest.raises(StopIteration):
        next(query)


@pytest.mark.parametrize(
    "restaurant_query",
    [
        [{"restaurant_name": "Andy's joy"}, {"restaurant_name": "Clare Hall"}],
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.parametrize(
    "state",
    [
        {"location": "Cambridge", "category": "Taiwanese"},
    ],
)
@pytest.mark.local
def test_search_command_state(restaurant_query, command_collection, state):
    query, entities, entity_name = restaurant_query
    first_entity = next(query)
    _ = query.__select__(entity=first_entity)
    query.state = state
    for slot, value in state.items():
        assert hasattr(query, slot)
        assert value == getattr(query, slot)

    recovered_state = query.state
    assert all(k in recovered_state for k in ("active_intent", "slot_values"))
    assert recovered_state["active_intent"] == query.name
    assert len(recovered_state["slot_values"]) == 3
    assert recovered_state["slot_values"].pop("restaurant_name") == ["Andy's joy"]
    assert recovered_state["slot_values"] == listify_values(state)
