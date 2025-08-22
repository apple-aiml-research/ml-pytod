#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.inference.constraint_prompts_utils import INVALID_SLOT, int2alpha
from pytod.inference.constraint_request import SlotConstraintRequest
from pytod.inference.schema_supervisor import SchemaSupervisor
from pytod.parser.expresion_validation_utils import ConstrainedKeyword

TEST_ON_SPLITS = ["test"]
VERSION = "v0.9.1"
data_version_params = {"split": TEST_ON_SPLITS, "version": VERSION}


constraint_request = [
    SlotConstraintRequest.model_validate(
        {
            "service": "Hotels_4",
            "predicted_slot": "hotel_name",
            "known_slots": (
                "stay_length",
                "number_of_rooms",
                "location",
                "star_rating",
            ),
        }
    )
]


@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.parametrize(
    "train_command_collection",
    [data_version_params],
    ids="split={}".format,
    indirect=True,
)
@pytest.mark.parametrize("constraint_request", constraint_request)
@pytest.mark.local
def test_hallucinated_arg_name_prompt(
    command_collection, train_command_collection, constraint_request
):
    agent = SchemaSupervisor(
        None, {}, command_collection, train_command_collection
    )
    agent.queue_request("1_00000", constraint_request)
    assert len(agent.queue) == 1
    indexed_slots = [
        s.strip().replace("- ", "").split(":")[0].split(" ")
        for s in agent.queue[0].prompt.split("\n")
        if s and s.strip().startswith("-")
    ]
    invalid_slot_token = indexed_slots[-1]
    assert agent.queue[0].slot_mapping.pop(
        invalid_slot_token[0][:1]
    ) == ConstrainedKeyword(INVALID_SLOT, None)
    assert (
        dict(
            [
                (pair[0][:1], ConstrainedKeyword(pair[1], None))
                for pair in indexed_slots[:-1]
            ]
        )
        == agent.queue[0].slot_mapping
    )


constraint_request = [
    SlotConstraintRequest.model_validate(
        {
            "service": "RentalCars_3",
            "predicted_slot": "pickup_date",
            "known_slots": (),
            "memorisation_info": {"services": ["RentalCars_1", "RentalCars_2"]},
        }
    )
]


@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.parametrize(
    "train_command_collection",
    [data_version_params],
    ids="split={}".format,
    indirect=True,
)
@pytest.mark.parametrize("constraint_request", constraint_request)
@pytest.mark.local
def test_memorised_arg_name_prompt(
    command_collection, train_command_collection, constraint_request
):
    agent = SchemaSupervisor(
        None, {}, command_collection, train_command_collection
    )
    agent.queue_request("1_00000", constraint_request)
    assert len(agent.queue) == 1
    prompt_lines = [
        s.strip().split(") ")[1].strip()
        for s in agent.queue[0].prompt.split("\n")
        if s and s.strip().startswith("-")
    ]
    descriptions, wildcard_line = prompt_lines[:-1], prompt_lines[-1]
    assert constraint_request.predicted_slot in wildcard_line
    slot_mapping = agent.queue[0].slot_mapping
    service = agent.queue[0].request.service
    for i, desc in enumerate(descriptions, start=1):
        arg_name = slot_mapping[int2alpha(i)].keyword
        try:
            assert (
                desc
                == command_collection.get_arg_schema(
                    service, arg_name
                ).description.lower()
            )
        except AssertionError:
            arg_schema = command_collection.get_arg_schema(service, arg_name)
            description = arg_schema.description.lower()
            description = (
                f"{description} ({', '.join(arg_schema.possible_values).lower()})"
            )
            try:
                assert desc == description
            except AssertionError:
                assert "(eg, " in desc
                assert sum([1 if v.lower() in desc else 0 for v in arg_schema.possible_values]) > 0
                assert arg_schema.description.lower() in desc
