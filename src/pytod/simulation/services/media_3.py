#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.pytod_types.aliases import DialogueID, SlotName
from pytod.simulation.command_registry import register_command
from pytod.simulation.confirmed_command import (
    ConfirmedCommand,
    ConfirmedCommandArgument,
)
from pytod.simulation.entities import Entity
from pytod.simulation.search_command import SearchCommand, SearchCommandArgument
from pytod.simulation.services_utils import __select__


@register_command(service="Media_3")
class FindMovies(SearchCommand):
    genre: SearchCommandArgument[str] = SearchCommandArgument()
    starring: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)

    def __select__(
        self,
        entity: Entity | None = None,
        key: SlotName | None = None,
        value: str | None = None,
    ) -> Entity | None:
        return __select__(self, entity, key, value)


@register_command(service="Media_3")
class PlayMovie(ConfirmedCommand):
    title: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    subtitle_language: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
