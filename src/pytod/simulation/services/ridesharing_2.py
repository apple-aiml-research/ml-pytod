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


@register_command(service="RideSharing_2")
class GetRide(ConfirmedCommand):
    destination: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    number_of_seats: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    ride_type: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
