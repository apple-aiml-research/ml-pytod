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


@register_command(service="Buses_1")
class FindBus(SearchCommand):
    from_location: SearchCommandArgument[str] = SearchCommandArgument()
    to_location: SearchCommandArgument[str] = SearchCommandArgument()
    leaving_date: SearchCommandArgument[str] = SearchCommandArgument()
    travelers: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Buses_1")
class BuyBusTicket(ConfirmedCommand):
    from_location: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    to_location: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    leaving_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    leaving_time: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    travelers: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
