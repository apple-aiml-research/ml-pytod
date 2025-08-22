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


@register_command(service="Events_3")
class FindEvents(SearchCommand):
    event_type: SearchCommandArgument[str] = SearchCommandArgument()
    city: SearchCommandArgument[str] = SearchCommandArgument()
    date: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Events_3")
class BuyEventTickets(ConfirmedCommand):
    event_name: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    number_of_tickets: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    city: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
