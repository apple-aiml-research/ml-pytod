#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

from pytod.pytod_types.aliases import SlotName


@dataclass
class Inform:
    dialog: str


class RecommendedAction(NamedTuple):
    dialog: str


@dataclass
class ActionResult:
    # index of the call that produced this result
    index: int
    result: Any = None
    recommended_action: Optional[list[RecommendedAction]] = None
    # name of the command that issued the result
    issuing_cmd: Optional[str] = None


@dataclass
class HintGenerator:
    slot_filling_template: str = (
        "Hint(ask the user to provide a value for: {slot_name})"
    )
    confirmation_template: str = "Hint(ask the user to confirm: {slot_name})"
    alternative_transaction_template: str = (
        "Hint(provide user with alternative: {slot_name})"
    )
    prompt_more_help_required_hint: str = (
        "Hint('ask the user if they require further assistance')"
    )

    def get_slot_filling_hint(self, slot_name: SlotName) -> str:
        return self.slot_filling_template.format(slot_name=slot_name)

    def get_confirmation_hint(self, slot_name: SlotName) -> str:
        return self.confirmation_template.format(slot_name=slot_name)

    def get_alternative_hint(self, slot_name: SlotName) -> str:
        return self.alternative_transaction_template.format(slot_name=slot_name)

    def prompt_user_hint(self) -> str:
        return self.prompt_more_help_required_hint
