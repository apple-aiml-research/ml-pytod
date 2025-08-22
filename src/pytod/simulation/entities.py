#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from dataclasses import field, make_dataclass
from typing import ClassVar, Dict, Optional, Protocol, Type, runtime_checkable

from pytod.interpreter.metadata import WILDCARD_VALUE
from pytod.pytod_types.aliases import CanonicalValue, SlotName

logger = logging.getLogger(__name__)

EntityDict = dict[SlotName, CanonicalValue]


@runtime_checkable
class Entity(Protocol):
    __dataclass_fields__: ClassVar[Dict]
    _identifier: list[str]
    _attributes: dict[str, str]


def get_entity(
    name: str, attributes: EntityDict, cmp_attributes: Optional[list[SlotName]] = None
) -> Entity:
    """A factory for entity instances.

    Parameters
    ----------
    name
        The class name of the entity returned.
    attributes
        The instance variables of the entity object.
    cmp_attributes
        The name of an attributes that are used to indentify to entities as identical.
        If not specified, all attributes are compared.
    """

    def get_entity_fields(
        attributes: EntityDict,
    ) -> list[tuple[str, Type[str] | Type[list], Type[field]]]:
        def attribs():
            return attributes

        fields = []
        for attrib, value in attributes.items():
            fields.append((attrib, str, field(default=value)))
        fields.append(("_attributes", dict[str, str], field(default_factory=attribs)))
        return fields

    def entity_comparator(self, other: Entity) -> bool:
        """Implementation of __eq__ for all entities."""
        properties = self._attributes
        # if the user states a wildcard value during search
        # the wildcard values of slots thare are not tracked
        # by the system is carried over to different queries
        # however, prior to v0.8.1 we reference the entity
        # property in the ground truth data so we carry-over
        # the property of the entity and not the wildcard value
        # this results in < 100% JGA. To counteract this, we
        # set the entity properties to "dontcare" but ensure
        # they cannot compare as equal so that we don't erroneously
        # raise StopIteration errors
        for obj in (self, other):
            for prop in self._identifier:
                if getattr(obj, prop, None) == WILDCARD_VALUE:
                    return False
        if self._identifier is not None:
            try:
                return all(
                    getattr(other, prop, None) == getattr(self, prop)
                    for prop in self._identifier
                )
            except AttributeError:
                return False
        for attribute, value in properties:
            if value != getattr(other, attribute, None):
                return False
        return True

    def __getattr__(self, name: str):
        msg = (
            f"Entity: {self.__class__.__name__}. "
            f"Attempted access to undefined property: '{name}'."
        )
        raise AttributeError(msg)

    fields = get_entity_fields(attributes)
    fields.append(("_identifier", list, field(default_factory=lambda: cmp_attributes)))
    namespace = {"__eq__": entity_comparator, "__getattr__": __getattr__}
    return make_dataclass(name, fields, namespace=namespace, slots=True)()
