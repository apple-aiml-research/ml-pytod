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


@register_command(service="Trains_1")
class FindTrains(SearchCommand):
    journey_starts_from: SearchCommandArgument[str] = SearchCommandArgument()
    to: SearchCommandArgument[str] = SearchCommandArgument()
    date_of_journey: SearchCommandArgument[str] = SearchCommandArgument()
    number_of_adults: SearchCommandArgument[str] = SearchCommandArgument()
    ticket_fare_class: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Trains_1")
class GetTrainTickets(ConfirmedCommand):
    journey_starts_from: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    to: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    date_of_journey: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    journey_start_time: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    number_of_adults: SearchCommandArgument[str] = ConfirmedCommandArgument()
    trip_protection: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    ticket_fare_class: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
