#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from pytod.interpreter.metadata import INTENT_UPDATE_TOOL, PARSE_ERROR_TOOL
from pytod.pytod_types.aliases import ToolName


class ConstrainedKeyword(NamedTuple):
    keyword: str
    value: str | None


@dataclass
class ParserFeedback:
    feedback: list[str] = field(default_factory=list)
    constraint: ConstrainedKeyword | None = None
    ignore_argument: bool = True
    pending_constraint: bool = False


def parse_error_dict() -> dict[str, Any]:
    return {
        "tool": PARSE_ERROR_TOOL,
        "keyword_args": [],
        "kwarg_values": [],
        "positional_args": [],
    }


def maybe_strip_object_reference(kwarg: str, tool_name: ToolName):
    """Remove object references from expression keyword."""
    kwarg_name = (
        kwarg[kwarg.find(".") + 1 :] if tool_name == INTENT_UPDATE_TOOL else kwarg
    )
    return kwarg_name


def contains_object_reference(kwarg_value: str) -> bool:
    return bool(re.match(r"^x\d{1,2}\.", kwarg_value))
