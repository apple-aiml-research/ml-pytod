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


@register_command(service="Banks_2")
class CheckBalance(SearchCommand):
    account_type: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Banks_2")
class TransferMoney(ConfirmedCommand):
    account_type: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    transfer_amount: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    recipient_name: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    recipient_account_type: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
