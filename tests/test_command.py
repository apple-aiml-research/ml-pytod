#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.simulation.command import Command


def test__getattr__():
    cmd = Command("1_00000")
    print(cmd.a)
    assert "a" in cmd._unknown_properties_accessed


def test__setattr__():
    cmd = Command("1_00000")
    cmd.a = 1
    setattr(cmd, "b", '2')
    assert "a" in cmd._unknown_properties_set
    cmd._entity_name = "Alex"
    assert cmd.name
    assert "name" not in cmd._unknown_properties_accessed
    assert cmd._unknown_properties_set == {"a": '1', "b": '2'}
