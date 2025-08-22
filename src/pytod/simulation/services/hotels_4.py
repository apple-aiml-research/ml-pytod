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


@register_command(service="Hotels_4")
class SearchHotel(SearchCommand):
    location: SearchCommandArgument[str] = SearchCommandArgument()
    smoking_allowed: SearchCommandArgument[str] = SearchCommandArgument()
    star_rating: SearchCommandArgument[str] = SearchCommandArgument()
    number_of_rooms: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Hotels_4")
class ReserveHotel(ConfirmedCommand):
    place_name: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    check_in_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    stay_length: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    location: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    number_of_rooms: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
