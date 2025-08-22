#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import functools
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from pytod.pytod_types.aliases import IntentName, ServiceName
from pytod.simulation.command import Command

logger = logging.getLogger(__name__)


def _registry_container():
    return defaultdict(dict)


@dataclass
class CommandRegistry:
    _registry: dict[ServiceName, dict[IntentName, Command]] = field(
        default_factory=_registry_container
    )

    def add(self, command: Command, service: ServiceName):
        name = command.__name__
        logger.debug(f"Registering object {command} under name {name}")
        self._registry[service][name] = command

    def get(self, *, name: IntentName, service: ServiceName, snake_cased: bool = False):
        if snake_cased:
            raise NotImplementedError
        return self._registry[service][name]


command_registry = CommandRegistry()


def register_command(service: ServiceName = ""):
    """A simple decorator that adds the commands
    to a registry available to the PyTOD agent."""

    def decorate(command: Command):
        if service:
            command_registry.add(command, service)
        else:
            raise ValueError("Service must be specified when registering a command")

        @functools.wraps(command)
        def wrapper(*args, **kwargs):
            return command

        return wrapper()

    return decorate
