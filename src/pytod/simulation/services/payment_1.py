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


@register_command(service="Payment_1")
class RequestPayment(ConfirmedCommand):
    receiver: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    amount: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    private_visibility: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Payment_1")
class MakePayment(ConfirmedCommand):
    receiver: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    amount: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    payment_method: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    private_visibility: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
