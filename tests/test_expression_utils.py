#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.parser.expressions import ExpressionList
from pytod.parser.expressions_utils import (
    extend_assignment,
    get_assignment_properties,
    map_to_assignment,
    replace_property,
)
from pytod.pytod_types.aliases import VariableName


@pytest.fixture
def simple_assignment() -> ExpressionList:
    return ExpressionList.from_string('x1.movie="Simone"')


@pytest.fixture()
def function_call() -> ExpressionList:
    return ExpressionList.from_string(
        "homes_2_find_home_by_area(area = 'san ramon', number_of_beds = 3)"
    )


@pytest.mark.parametrize(
    "member, expected_output",
    [
        (("device", '"TV"'), "x1.movie = 'Simone'; x1.device = \"TV\""),
    ],
)
def test_extend_assignment(
    simple_assignment: ExpressionList, member: tuple[str, str], expected_output: str
):
    slot, value = member
    extend_assignment(simple_assignment, slot, value)
    assert str(simple_assignment) == expected_output


@pytest.mark.parametrize(
    "member, expected_output",
    [
        (("movie", "name"), "x1.name = 'Simone'"),
    ],
)
def test_replace_property(
    simple_assignment: ExpressionList, member: tuple[str, str], expected_output: str
):
    src_slot, tgt_slot = member
    replace_property(simple_assignment, src_slot, tgt_slot)
    assert str(simple_assignment) == expected_output


@pytest.mark.parametrize("expected_output", [{"movie": "'Simone'"}])
def test_get_assignment_properties(
    simple_assignment: ExpressionList, expected_output: dict[str, str]
):
    assert expected_output == get_assignment_properties(simple_assignment)


@pytest.mark.parametrize(
    "variable, expected_output",
    [
        ("x0", "x0.area = 'san ramon'; x0.number_of_beds = 3"),
    ],
)
def test_map_to_assignment(
    function_call: ExpressionList, variable: VariableName, expected_output: str
):
    assert str(map_to_assignment(function_call, variable)) == expected_output
