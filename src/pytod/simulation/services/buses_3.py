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


@register_command(service="Buses_3")
class FindBus(SearchCommand):
    from_city: SearchCommandArgument[str] = SearchCommandArgument()
    to_city: SearchCommandArgument[str] = SearchCommandArgument()
    departure_date: SearchCommandArgument[str] = SearchCommandArgument()
    num_passengers: SearchCommandArgument[str] = SearchCommandArgument()
    category: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Buses_3")
class BuyBusTicket(ConfirmedCommand):
    from_city: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    to_city: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    departure_time: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    departure_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    num_passengers: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    additional_luggage: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
