#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from unittest.mock import patch

import pytest

from pytod.inference.parser_supervisor import (
    ProcessedSupervisionRequest,
    SupervisionRequest,
)
from pytod.prompting.pytod_nlu_formatters import UNK_VALUE


@pytest.fixture
def mock_supervision_request() -> SupervisionRequest:
    """Fixture to create a mock SupervisionRequest."""
    return SupervisionRequest(
        prompt="",
        service="",
        turn_idx=2,
        dialogue_id="",
        parser_map={1: "slot1", 2: "slot2", 3: "slot3", 4: "slot4", 5: "slot5"},
    )


@pytest.mark.parametrize(
    "pred_str,expected_output",
    [
        ("1) san jose 2) unanswerable", {"slot1": "san jose", "slot2": UNK_VALUE}),
        ("1) new york 2) california", {"slot1": "new york", "slot2": "california"}),
        (
            "1) 11:30 2) saratoga 3) sipan 4) 1 5) 5th of march",
            {
                "slot1": "11:30",
                "slot2": "saratoga",
                "slot3": "sipan",
                "slot4": "1",
                "slot5": "5th of march",
            },
        ),
        ("", None),
    ],
)
def test_prediction_parsing(
    mock_supervision_request: SupervisionRequest,
    pred_str: str,
    expected_output: dict[str, str] | None,
):
    # Initialize ProcessedSupervisionRequest with the fixture
    processed_request = ProcessedSupervisionRequest(request=mock_supervision_request)

    # Set the prediction
    processed_request.prediction = pred_str

    # Assert the parsed predictions match expected output
    assert processed_request.prediction == expected_output


@pytest.mark.parametrize("pred_str", ("7) invalid_answer",))
def test_invalid_index_logging(
    mock_supervision_request: SupervisionRequest, pred_str: str
):
    with patch("pytod.inference.parser_supervisor.logger") as mock_logger:
        processed_request = ProcessedSupervisionRequest(
            request=mock_supervision_request
        )

        processed_request.prediction = pred_str

        # Assert that the logger caught the error
        mock_logger.error.assert_called_once_with(
            f"{processed_request.request}: Supervisor output an invalid index 7. "
            f"Parser map: {mock_supervision_request.parser_map}"
        )
        assert processed_request.prediction is None


@pytest.mark.parametrize("pred_str", ("1) unanswerable",))
def test_warning_on_unknown_value(
    mock_supervision_request: SupervisionRequest, pred_str: str
):
    with patch("pytod.inference.parser_supervisor.logger") as mock_logger:
        processed_request = ProcessedSupervisionRequest(
            request=mock_supervision_request
        )
        processed_request.prediction = pred_str
        mock_logger.warning.assert_called_once_with(
            f"{processed_request.request} Supervisor failed to parse slot slot1"
        )
        assert processed_request.prediction == {"slot1": UNK_VALUE}
