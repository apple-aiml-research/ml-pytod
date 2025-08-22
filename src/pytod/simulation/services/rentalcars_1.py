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


@register_command(service="RentalCars_1")
class GetCarsAvailable(SearchCommand):
    pickup_city: SearchCommandArgument[str] = SearchCommandArgument()
    pickup_date: SearchCommandArgument[str] = SearchCommandArgument()
    dropoff_date: SearchCommandArgument[str] = SearchCommandArgument()
    pickup_time: SearchCommandArgument[str] = SearchCommandArgument()
    type: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="RentalCars_1")
class ReserveCar(ConfirmedCommand):
    pickup_location: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    pickup_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    pickup_time: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    dropoff_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    type: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
