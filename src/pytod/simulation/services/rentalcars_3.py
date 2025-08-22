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


@register_command(service="RentalCars_3")
class GetCarsAvailable(SearchCommand):
    city: SearchCommandArgument[str] = SearchCommandArgument()
    start_date: SearchCommandArgument[str] = SearchCommandArgument()
    end_date: SearchCommandArgument[str] = SearchCommandArgument()
    pickup_time: SearchCommandArgument[str] = SearchCommandArgument()
    car_type: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="RentalCars_3")
class ReserveCar(ConfirmedCommand):
    pickup_location: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    start_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    pickup_time: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    end_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    car_type: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    add_insurance: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
