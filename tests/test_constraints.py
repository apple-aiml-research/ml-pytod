#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.inference.constraint_request import SlotConstraintRequest
from pytod.inference.schema_supervisor import ProcessedConstraintRequest
from pytod.parser.expresion_validation_utils import ConstrainedKeyword


def test_slot_constraints_hash():
    s1 = {"predicted_slot": "stay_long", "known_slots": ("a", "b"), "service": "2"}
    s2 = {"predicted_slot": "stay_long", "known_slots": ("a",), "service": "1"}
    s3 = {
        "predicted_slot": "stay_long",
        "known_slots": ("a",),
        "predicted_slot_value": "x",
        "service": "1",
    }

    slots = set()
    for s in (s1, s2, s3):
        slots.add(SlotConstraintRequest.model_validate(s))
    assert len(slots) == 2
    pc_dict = {
        "prompt": "x",
        "session_id": "1_00000",
        "request": list(slots)[0],
        "slot_mapping": {"x": ConstrainedKeyword("y", None)},
    }
    pc = ProcessedConstraintRequest.model_validate(pc_dict)
    pc.selected_option = "x"
    assert pc.constrained_keyword.keyword == "y"


def test_pending_constraints():
    constraint = SlotConstraintRequest.model_validate(
        {"predicted_slot": "stay_long", "known_slots": ("a", "b"), "service": "2"}
    )
    pending_constraint = ProcessedConstraintRequest.model_validate(
        {
            "prompt": "x",
            "session_id": "1_00000",
            "request": constraint,
            "slot_mapping": {"a": ConstrainedKeyword("my_favourite_slot", None)},
        }
    )
    pending_constraint.selected_option = "a"
    assert pending_constraint.constrained_keyword.keyword == "my_favourite_slot"
