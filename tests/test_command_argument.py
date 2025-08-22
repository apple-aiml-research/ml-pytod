#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

TEST_ON_SPLITS = ["dev"]
VERSION = "v0.9.1"

data_version_params = {"split": TEST_ON_SPLITS, "version": VERSION}


@pytest.mark.parametrize(
    "restaurant_query",
    [
        [{"restaurant_name": "A"}, {"restaurant_name": "B"}],
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "command_collection", [data_version_params], ids="split={}".format, indirect=True
)
@pytest.mark.local
def test__set__(restaurant_query, command_collection):
    query, _, _ = restaurant_query
    # check the categorical casing is changed as expected
    query.has_seating_outdoors = "true"
    query.location = "Seattle"
    assert query.has_seating_outdoors == "True"
    assert query.location == "Seattle"
