#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.simulation.database import dev_databases, test_databases


@pytest.mark.parametrize(
    "database_split",
    [
        (dev_databases, "dev"),
        (test_databases, "test"),
    ],
    ids=lambda x: x[1],
)
@pytest.mark.local
def test_database_init(database_split):
    """Check databases are initialised."""
    database, split = database_split
    for service in database:
        for intent in database[service]:
            this_intent_db = database[service][intent]
            assert this_intent_db.num_items > 0
            assert this_intent_db.split == split
