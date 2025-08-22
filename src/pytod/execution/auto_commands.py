#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Optional

from pytod.simulation.confirmed_command import ConfirmedCommand


@dataclass
class Signal:
    dialog: str


class AutoCommand:
    pass


class Perform(AutoCommand):
    """Marks the successful call to a transactional
    intent."""

    args: Optional[ConfirmedCommand] = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["args"]


class Hint(AutoCommand):
    """A recommendation coming from a simulated SGD app.
    app.

    Args:
        message: the hint to follow.
        ref: the reference to the variable representing the task.
    """

    message: Optional[str] = None
    ref: Optional[str] = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["message", "ref"]
