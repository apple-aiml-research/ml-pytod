#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional

from pytod.pytod_types.aliases import DialogueID, SlotName
from pytod.simulation.command_registry import register_command
from pytod.simulation.confirmed_command import (
    ConfirmedCommand,
    ConfirmedCommandArgument,
)
from pytod.simulation.entities import Entity
from pytod.simulation.search_command import SearchCommand, SearchCommandArgument
from pytod.simulation.services_utils import __select__


@register_command(service="Media_2")
class FindMovies(SearchCommand):
    genre: SearchCommandArgument[str] = SearchCommandArgument()
    actors: SearchCommandArgument[str] = SearchCommandArgument()
    director: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)

    def __select__(
        self,
        entity: Optional[Entity] = None,
        key: Optional[SlotName] = None,
        value: Optional[str] = None,
    ) -> Optional[Entity]:
        return __select__(self, entity, key, value)


@register_command(service="Media_2")
class RentMovie(ConfirmedCommand):
    movie_name: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    subtitle_language: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
