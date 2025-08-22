#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Utilities used for custom service policy implementation."""
import logging

from pytod.pytod_types.aliases import SlotName
from pytod.simulation._command_utils import EntitySelectionError, SearchError
from pytod.simulation.entities import Entity, get_entity

logger = logging.getLogger(__name__)


def __select__(
    self,
    entity: Entity | None = None,
    key: SlotName | None = None,
    value: str | None = None,
) -> Entity | None:
    """Custom select implementation for:

    - Media_2, FindMovies
    """
    if key is not None:
        try:
            assert (
                value
            ), f"Selected entity by specifying key={key} but value=None. Expected value."
        except AssertionError as e:
            raise EntitySelectionError(e.args[0])
        if (
            self.current_entity is not None
            and getattr(self.current_entity, key) == value
        ):
            self.selected_entity = self.current_entity
            return self.current_entity
        else:
            normalisation_result = self._normalizer.normalise(
                self.service, {key: value}
            ).result
            if normalisation_result:
                normalised_value = normalisation_result[key].lower()
            else:
                normalised_value = value
            for entity in self._raw_entities:
                if entity[key] == value or entity[key] == normalised_value:
                    entity_obj = get_entity(
                        self.entity_name, entity, cmp_attributes=self._entity_cmp_key
                    )
                    self.selected_entity = entity_obj
                    return entity_obj
            raise SearchError(f"No mentioned entity matched {key}={value}.")
    if entity is not None:
        self.selected_entity = entity
