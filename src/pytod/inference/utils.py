#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import re
from dataclasses import dataclass
from typing import Optional

from fuzzywuzzy import fuzz

from pytod.pytod_types.aliases import IntentName, ServiceName
from pytod.utils import snake_to_camel


def get_lineno(prompt: str) -> int:
    """Extract the current line number from the prompt."""
    try:
        lineno = int(prompt[-2:])
    except ValueError:
        lineno = int(prompt[-1:])
    return lineno


def map_tool_name_to_service_intent(
    tool_name: str,
) -> tuple[Optional[ServiceName], IntentName]:
    """Recover the service and intent name from
    `tool_name`(format [{service_name}_]{intent_name})."""
    service, intent_name = None, tool_name
    service_pattern = r"^\w+_\d+"
    match = re.search(service_pattern, tool_name)
    if match is not None:
        *name_pieces, service_id = match.group(0).split("_")
        service_name = "".join([p.capitalize() for p in name_pieces])
        service = f"{service_name}_{service_id}"
        intent_name = snake_to_camel(tool_name[len(service) + len(name_pieces) :])
    return service, intent_name


def fuzzy_string_match(str_ref: str, str_hyp: str):
    """Returns fuzzy string similarity score in range [0.0, 1.0]."""

    # The higher the score, the higher the similarity between the two strings.
    return fuzz.token_sort_ratio(str_ref, str_hyp) / 100.0


@dataclass
class FuzzyMatch:
    match: str
    score: float


def find_closest_match(to_match: str, alternatives: list[str]) -> FuzzyMatch:
    best_score = 0.0
    closest_match = None
    for alternative in alternatives:
        score = fuzzy_string_match(alternative, to_match)
        if score > best_score:
            closest_match = alternative
            best_score = score
    return FuzzyMatch(match=closest_match, score=best_score)


def var_to_index(var: str) -> int:
    assert (
        m := re.match(VARIABLE_PATTERN, var)
    ) is not None, f"Illegal variable name: {var}"
    return int(m["ndx"])


VARIABLE_PATTERN = r"x(?P<ndx>[0-9]+)"
