#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

TEST_ON_SPLITS = ["test"]
VERSION = "v0.9.1"
data_version_params = {"split": TEST_ON_SPLITS, "version": VERSION}


@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.local
def test_service_arg_definitions(command_collection):
    arg_defs = command_collection.get_service_arg_definitions("Restaurants_2")
    assert len(arg_defs) == 9
