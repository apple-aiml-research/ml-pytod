#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import pytest

from pytod.simulation.database_utils import Normalizer


@pytest.mark.parametrize(
    "parameters",
    [
        (
            {
                "service": "Hotels_2",
                "input": {
                    "where_to": "LAX",
                    "check_out_date": "march 12th",
                    "check_in_date": "2019-03-11",
                },
                "expected_output": {
                    "where_to": "Los Angeles",
                    "check_out_date": "2019-03-12",
                    "check_in_date": "2019-03-11",
                },
            }
        ),
        (
            {
                "service": "Hotels_2",
                "input": {
                    "where_to": "lax",
                    "check_out_date": "march 12th",
                    "check_in_date": "2019-03-11",
                },
                "expected_output": {
                    "where_to": "Los Angeles",
                    "check_out_date": "2019-03-12",
                    "check_in_date": "2019-03-11",
                },
            }
        ),
    ],
)
@pytest.mark.local
def test_normalizer(parameters):
    normalizer = Normalizer()
    normalised_params = normalizer.normalise(parameters["service"], parameters["input"])
    assert normalised_params.result == parameters["expected_output"]
