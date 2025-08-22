#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import os
from pathlib import Path
from typing import Mapping, Optional

from pydantic import BaseModel

from pytod.pytod_types.aliases import CanonicalValue, ServiceName, SlotName

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    pass


def hash_query_result(
    result: dict[SlotName, CanonicalValue]
) -> tuple[tuple[SlotName, CanonicalValue]]:
    """Create a hashable object from a dict containing entity information."""
    return tuple(sorted(list(result.items()), key=lambda x: x[0]))


def serialise_result(result: dict[SlotName, CanonicalValue]) -> str:
    """Create a `str` object from an SGD database entry."""
    return json.dumps(hash_query_result(result))


def deserialise_result(result_str: str) -> dict[SlotName, CanonicalValue]:
    """Create a `dict` object from a serialised SGD database entry."""
    return dict(json.loads(result_str))


class NormalisedCallParameters(BaseModel):
    result: dict[SlotName, CanonicalValue | str]
    # slots for which normalisation_orig was not possible
    failed_normalisation: Optional[dict[SlotName, str]] = None


class Normalizer:
    """A class for mapping argument values extracted from conversation
    to their canonical form."""

    def __init__(self):
        if os.getenv("NORMALISATION_LOOKUP") is not None:
            lookup_pth = Path(os.getenv("NORMALISATION_LOOKUP"))
            with open(lookup_pth, "r") as f:
                equivalence_map = json.load(f)
            self._equivalence_map = equivalence_map
        else:
            logger.warning(
                "Could not find normalisation lookup table. Parameters will not "
                "be normalised. Have you run setup.sh?"
            )
            self._equivalence_map = None

    def _search_map(self, arg_value: str) -> str | None:
        """Search the slot value to be normalised in the normalisation table of
        the other services if a given value could not be normalised. This saves
        us for manually including carry-over values in the normalisation map.

        Example
        -------
        The value "fresno, ca" cannot be found in the normalisation table of `Hotels_4`
        but will be found in the normalisation table of `Buses_1`, from where it is
        inherited.
        """
        for service in self._equivalence_map:
            for arg in self._equivalence_map[service]:
                if arg_value in (
                    arg_canonical_vals := self._equivalence_map[service][arg]
                ):
                    return arg_canonical_vals[arg_value]

    def _normalise_arg(
        self, arg: SlotName, arg_value: str, service: ServiceName
    ) -> CanonicalValue | None:
        """Normalise the value of slot `arg` from service `service`."""
        try:
            arg_value = self._equivalence_map[service][arg][arg_value]
            return arg_value
        except KeyError:
            try:
                if arg_value in self._equivalence_map[service][arg].values():
                    return arg_value
            except KeyError:
                # raised if a slot is hallucinated
                pass
            # we replace double quotes with singles but the normalisation
            # table is extracted from the raw data, so these values may
            # appear not to be in the normalisation table
            if "'" in arg_value:
                try:
                    arg_value = self._equivalence_map[service][arg][
                        arg_value.replace("'", '"')
                    ]
                    return arg_value
                except KeyError:
                    self._search_map(arg_value)
            return self._search_map(arg_value)

    def normalise(
        self, service: ServiceName, arguments: Mapping[str, str | int | bool]
    ) -> NormalisedCallParameters:
        """Convert parameters extracted from conversation to their canonical
        form prior to database calls."""
        if self._equivalence_map is None:
            return NormalisedCallParameters.model_validate(
                {
                    "result": arguments,
                },
            )
        failed_normalisation = {}
        normalised_params = {}
        for arg, arg_value in arguments.items():
            arg_value = str(arg_value)
            normalised_value = self._normalise_arg(arg, arg_value, service)
            if normalised_value is None:
                failed_normalisation[arg] = arg_value
            normalised_params[arg] = (
                arg_value if normalised_value is None else normalised_value
            )
        result_dict = {"result": normalised_params}
        if failed_normalisation:
            result_dict.update({"failed_normalisation": failed_normalisation})
        return NormalisedCallParameters.model_validate(result_dict)
