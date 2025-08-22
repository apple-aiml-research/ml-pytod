#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.simulation.api_driver import dev_apis, test_apis


@pytest.mark.parametrize(
    "apis_split",
    [
        (dev_apis, "dev"),
        (test_apis, "test"),
    ],
    ids=lambda x: x[1],
)
@pytest.mark.local
def test_apis_init(apis_split):
    """Check apis are initialised."""
    apis, split = apis_split
    for service in apis:
        for intent in apis[service]:
            this_api = apis[service][intent]
            assert len(this_api._collection) > 0
            assert this_api._split == split
