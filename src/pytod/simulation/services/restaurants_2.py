#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.pytod_types.aliases import DialogueID
from pytod.simulation.command_registry import register_command
from pytod.simulation.confirmed_command import (
    ConfirmedCommand,
    ConfirmedCommandArgument,
)
from pytod.simulation.search_command import SearchCommand, SearchCommandArgument


@register_command(service="Restaurants_2")
class FindRestaurants(SearchCommand):
    location: SearchCommandArgument[str] = SearchCommandArgument()
    category: SearchCommandArgument[str] = SearchCommandArgument()
    price_range: SearchCommandArgument[str] = SearchCommandArgument()
    has_seating_outdoors: SearchCommandArgument[str] = SearchCommandArgument()
    has_vegetarian_options: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Restaurants_2")
class ReserveRestaurant(ConfirmedCommand):
    restaurant_name: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    location: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    time: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    number_of_seats: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
