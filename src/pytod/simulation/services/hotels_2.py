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


@register_command(service="Hotels_2")
class BookHouse(ConfirmedCommand):
    where_to: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    check_in_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    check_out_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    number_of_adults: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Hotels_2")
class SearchHouse(SearchCommand):
    where_to: SearchCommandArgument[str] = SearchCommandArgument()
    has_laundry_service: SearchCommandArgument[str] = SearchCommandArgument()
    number_of_adults: SearchCommandArgument[str] = SearchCommandArgument()
    rating: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
