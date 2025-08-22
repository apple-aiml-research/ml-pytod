#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.simulation.command import SlotState
from pytod.simulation.confirmed_command import ConfirmedCommand
from pytod.simulation.search_command import SearchCommand

TEST_ON_SPLITS = ["dev"]
VERSION = "v0.9.1"

data_version_params = {"split": TEST_ON_SPLITS, "version": VERSION}


@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.local
def test_build(command_collection):
    reserve_restaurant_cmd = command_collection.get(
        "Restaurants_2", "ReserveRestaurant"
    )

    reserve_restaurant = ConfirmedCommand.build("1_00000", reserve_restaurant_cmd)
    assert isinstance(reserve_restaurant, ConfirmedCommand)
    assert not isinstance(reserve_restaurant, SearchCommand)
    assert reserve_restaurant.positional_args == ("restaurant_name", "location", "time")
    assert reserve_restaurant.keyword_args == {
        "number_of_seats": "2",
        "date": "2019-03-01",
    }
    assert reserve_restaurant._args_for_confirmation == (
        "date",
        "location",
        "number_of_seats",
        "restaurant_name",
        "time",
    )
    assert reserve_restaurant.entity_name == "RestaurantReservation"
    assert reserve_restaurant.followup_command is None
    assert reserve_restaurant.service == "Restaurants_2"


@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.local
def test__confirm__(command_collection):
    reserve_restaurant_cmd = command_collection.get(
        "Restaurants_2", "ReserveRestaurant"
    )

    reserve_restaurant = ConfirmedCommand.build("1_00000", reserve_restaurant_cmd)
    reserve_restaurant.__confirm__()
    assert reserve_restaurant._user_confirmed
    for arg, arg_state in reserve_restaurant._keywords_state.items():
        assert arg_state == SlotState.SET
