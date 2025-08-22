#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from copy import deepcopy
from enum import Enum
from typing import Any


class APICallStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


def deffer_to_descriptor(setattr_func):
    def setattr_wrapper(self, attr, value):
        if attr in self._descriptors:
            return object.__setattr__(self, attr, value)
        return setattr_func(self, attr, value)

    return setattr_wrapper


class _CommandSetattrControl(type):
    """A metaclass that::

    - marks attributes which implement the descriptor protocol

    - introspects __init__ to find out the name of the instance
    variables, populating the `_instance_attributes` class variable

    - wraps the __setattr__ method of the object such that
    the execution is deferred to the __set__ when a property for
    which a descriptor exists is set
    """

    def __new__(metacls, name: str, bases: tuple, dct: dict[str, Any]):
        def get_instance_attributes(obj):
            return {el for el in obj.__code__.co_names if el not in dir(obj)}

        def get_bases_instance_attributes(bases):
            if not bases:
                return set()
            obj = bases[0]
            match obj.__name__:
                case "Command" | "SearchCommand" | "ConfirmedCommand":
                    for key, obj_ in obj.__dict__.items():
                        if key == "__init__":
                            return get_instance_attributes(obj_)
                case _:
                    return set()

        try:
            descriptors = deepcopy(bases[0].__dict__["_descriptors"])
        except KeyError:
            descriptors = set()
        except IndexError:
            descriptors = set()
        instance_attribs = get_bases_instance_attributes(bases)
        try:
            instance_attribs = instance_attribs.union(
                bases[0].__dict__["_instance_attributes"]
            )
        except KeyError:
            pass
        except IndexError:
            instance_attribs = set()

        for key, obj in dct.items():
            if key == "__init__":
                instance_attribs.update(get_instance_attributes(obj))
            if key == "__setattr__":
                dct[key] = deffer_to_descriptor(obj)
            elif hasattr(obj, "__get__"):
                descriptors.add(key)
        dct["_instance_attributes"] = frozenset(instance_attribs)
        dct["_descriptors"] = descriptors
        return type.__new__(metacls, name, bases, dct)


class SearchError(Exception):
    pass


class EntitySelectionError(Exception):
    pass
