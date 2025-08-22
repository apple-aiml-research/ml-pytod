#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from functools import partial

import pytest

from pytod.parser.expresion_validation_context import SchemaConstraintContext
from pytod.parser.expressions import ExpressionList, expression_init_context
from pytod.toolbox.toolbox import get_command_name

TEST_ON_SPLITS = ["dev"]
VERSION = "v0.9.1"


@pytest.fixture
def tool_name_formatter(request):
    use_snake_case = request.param["use_snake_case"]
    skip_service_variations = request.param["skip_service_variations"]
    return partial(
        get_command_name,
        use_snake_case=use_snake_case,
        skip_service_variations=skip_service_variations,
    )


tool_formatter_params = {"use_snake_case": True, "skip_service_variations": False}


@pytest.fixture
def instructions():
    return [
        {
            "input": "select(x15, from_results=x)",
            "expected": "select(x15)",
            "intent_predicted": False,
        },
        {
            "input": "select(x15, from_results = x11)",
            "expected": "select(x15, from_results = x11)",
            "intent_predicted": False,
        },
        {
            "input": "select()",
            "expected": "select()",
            "intent_predicted": False,
        },
        {
            "input": "alarms_1_add_alarm()",
            "expected": "alarm_1_add_alarm()",
            "intent_predicted": True,
        },
        {
            "input": "rms_1_add_alarm()",
            "expected": "alarm_1_add_alarm()",
            "intent_predicted": True,
        },
        {
            "input": "alarm_1_add_alarm()",
            "expected": "alarm_1_add_alarm()",
            "intent_predicted": True,
        },
        {
            "input": 'alarm_1_add_alarm(location="Cambridge")',
            "expected": "alarm_1_add_alarm()",
            "intent_predicted": True,
        },
        {
            "input": 'alarma_1_add_alarm(location="Cambridge")',
            "expected": "alarm_1_add_alarm()",
            "intent_predicted": True,
        },
        {
            "input": 'alarm_1_add_alarm(new_alarm_name="x", new_alarm_nam="y")',
            "expected": "alarm_1_add_alarm(new_alarm_name = 'x')",
            "intent_predicted": True,
        },
        {
            "input": "alarms_1_add_alarm(new_alarm_tim='17:50', not_arg='bad')",
            "expected": "alarm_1_add_alarm(new_alarm_time = '17:50')",
            "intent_predicted": True,
        },
        {
            "input": "x1.new_alarm_time = '17:50'",
            "expected": "x1.new_alarm_time = '17:50'",
            "intent_predicted": False,
        },
        {
            "input": "x1.new_alarm_time = '17:50'; x1.new_alarm_nam = 'finish wor'",
            "expected": "x1.new_alarm_time = '17:50'; x1.new_alarm_name = 'finish wor'",
            "intent_predicted": False,
        },
        {"input": "x5.a = b", "expected": "parse_error()", "intent_predicted": False},
        {
            "input": "x5.a = b; x1.new_alarm_time='11:00'",
            "expected": "x1.new_alarm_time = '11:00'",
            "intent_predicted": False,
        },
        {
            "input": "x1.location = b; x1.new_alarm_time='11:00'",
            "expected": "x1.new_alarm_time = '11:00'",
            "intent_predicted": False,
        },
        {
            "input": "x1.location = b; x1.new_alarm_time='11:00'",
            "expected": "x1.new_alarm_time = '11:00'",
            "intent_predicted": False,
        },
        {
            "input": "hot_1_add_available(new_provider_name = 'phone home')",
            "expected": "alarm_1_add_alarm()",
            "intent_predicted": True,
        },
        # fails because of a change to
        # {
        #     "input": "restaurants_2_find_restaurants(restaurant_name = 'xx', z=t)",
        #     "expected": "restaurants_2_find_restaurants(restaurant_name = 'xx')",
        #     "intent_predicted": True
        # },
    ]


data_version_params = {"split": TEST_ON_SPLITS, "version": VERSION}


@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.parametrize("tool_name_formatter", [tool_formatter_params], indirect=True)
@pytest.mark.local
def test_calls_constraints(command_collection, tool_name_formatter, instructions):
    context = {
        "dial_id": "unittest",
        "schema": command_collection,
        "last_user_turn_service": "Alarm_1",
        "variable_mapping": {1: "alarm_1_add_alarm"},
        "intent_predicted": False,
    }
    context = SchemaConstraintContext(**context)
    for instr in instructions:
        # context["intent_predicted"] = False
        with expression_init_context(context):
            expr = ExpressionList.from_string(instr["input"])
            actual = str(expr)
        assert actual == instr["expected"]
        if instr["input"] == instr["expected"]:
            assert not expr.feedback
        else:
            assert expr.feedback
