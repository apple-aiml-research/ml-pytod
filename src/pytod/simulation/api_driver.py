#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Literal, Mapping, NamedTuple, Optional, TypeVar

from mongoquery import Query

from pytod.pytod_types.aliases import IntentName, ServiceName, SlotName, SlotValue
from pytod.simulation._command_utils import APICallStatus

T = TypeVar("T", bound=Mapping)

logger = logging.getLogger(__name__)


class TransactionResult(NamedTuple):
    status: APICallStatus
    alternative: Optional[dict[SlotName, list[SlotValue]]] = None
    response: Optional[dict[str, str]] = None


class APIDriver:
    def __init__(self, collection: list[T], split: Literal["dev", "test"]):
        self._collection = collection
        self._split = split

    @staticmethod
    def _lowercase_params(
        parameters: Mapping[SlotName, SlotValue]
    ) -> dict[SlotName, SlotValue]:
        """The API responses collection is lowercased, so lower-casing is
        necessary to ensure the API calls return the correct results."""
        return {p: parameters[p].lower() for p in parameters}

    def __call__(self, parameters: Mapping[SlotName, SlotValue]) -> TransactionResult:
        """Call the API with `parameters`."""
        parameters = self._lowercase_params(deepcopy(parameters))
        q = Query(parameters)
        response = [deepcopy(item) for item in self._collection if q.match(item)]
        if not response:
            dial_id = parameters["dialogue_id"]
            logger.warning(
                f"{dial_id}: Parameters {parameters} did not match the API schema"
            )
            return TransactionResult(status=APICallStatus.FAILURE, alternative=None)
        response = response[0]
        response.pop("dialogue_id")
        match response["status"]:
            case "SUCCESS":
                return TransactionResult(
                    status=APICallStatus.SUCCESS, alternative=None, response=response
                )
            case "FAILURE":
                return TransactionResult(
                    status=APICallStatus.FAILURE,
                    alternative=response["alternative"],
                    response=response,
                )


def initialise_apis(
    split: Literal["dev", "test"]
) -> Optional[dict[ServiceName, dict[IntentName, APIDriver]]]:
    """Initialise APIs necessary for simulating responses of transactional APIs."""

    def init_api(
        responses: list[
            dict[
                SlotName | Literal["status", "alternative", "dialogue_id"],
                str | list[SlotValue],
            ]
        ],
        split: Literal["dev", "test"],
        lowercase: bool = True,
    ) -> APIDriver:
        cased_responses = []
        if lowercase:
            for r in responses:
                for attrib, value in r.items():
                    if attrib not in ["status", "alternative", "dialogue_id"]:
                        r[attrib] = value.lower()
                cased_responses.append(r)

        return APIDriver(responses if not cased_responses else cased_responses, split)

    assert split in ["dev", "test"], f"No database definition for split {split}"
    if os.getenv("API_RESPONSES") is not None:
        api_responses = Path(os.getenv("API_RESPONSES")) / f"{split}.json"
    else:
        logger.error(
            "Could not initialise databases because API_RESPONSES env var was not set."
            "Have you run setup.sh?"
        )
        return
    with open(api_responses, "r") as f:
        responses = json.load(f)

    servers = {service: {} for service in responses}
    for service in servers:
        for intent, this_intent_responses in responses[service].items():
            servers[service][intent] = init_api(this_intent_responses, split)
    return servers


dev_apis = initialise_apis("dev")
test_apis = initialise_apis("test")
