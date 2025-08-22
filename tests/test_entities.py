#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.simulation.entities import Entity, get_entity


def test_get_entity():
    attributes = {
        "restaurant_name": "Nando's",
        "location": "Cambridge",
        "cuisine": "international",
    }

    restaurant = get_entity(
        "Restaurant", attributes=attributes, cmp_attributes=["restaurant_name"]
    )
    assert restaurant.restaurant_name == "Nando's"
    assert restaurant.location == "Cambridge"
    assert restaurant.cuisine == "international"
    assert isinstance(restaurant, Entity)
    my_entity_list = [restaurant]
    other = get_entity(
        "Restaurant",
        {
            "restaurant_name": "Restaurant 22",
            "location": "Cambridge",
            "cuisine": "international",
        },
    )
    other_nandos = get_entity(
        "Restaurant",
        {"restaurant_name": "Nando's", "location": "Dublin", "cuisine": "chicken"},
    )
    assert other not in my_entity_list
    assert other_nandos in my_entity_list
