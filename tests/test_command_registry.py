#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.simulation.confirmed_command import ConfirmedCommand
from pytod.simulation.search_command import SearchCommand


@pytest.mark.parametrize(
    "service_command",
    [["Restaurants_2", "FindRestaurants"]],
    ids=lambda x: f"service={x[0]}, intent={x[1]}",
)
@pytest.mark.local
def test_command_registry(service_command):
    from pytod.simulation.command_registry import command_registry

    assert command_registry
    service, intent = service_command
    find_restaurant = command_registry.get(name=intent, service=service)
    find_restaurant_instance = find_restaurant.build("id")
    assert isinstance(find_restaurant_instance, SearchCommand)
    assert not isinstance(find_restaurant_instance, ConfirmedCommand)
    assert find_restaurant.__name__ == intent
