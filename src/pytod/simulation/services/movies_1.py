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


@register_command(service="Movies_1")
class FindMovies(SearchCommand):
    location: SearchCommandArgument[str] = SearchCommandArgument()
    theater_name: SearchCommandArgument[str] = SearchCommandArgument()
    genre: SearchCommandArgument[str] = SearchCommandArgument()
    show_type: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)

    def __select__(
        self,
        entity: Entity | None = None,
        key: SlotName | None = None,
        value: str | None = None,
    ) -> Entity | None:
        return __select__(self, entity, key, value)


@register_command(service="Movies_1")
class GetTimesForMovie(SearchCommand):
    location: SearchCommandArgument[str] = SearchCommandArgument()
    movie_name: SearchCommandArgument[str] = SearchCommandArgument()
    show_date: SearchCommandArgument[str] = SearchCommandArgument()
    theater_name: SearchCommandArgument[str] = SearchCommandArgument()
    show_type: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Movies_1")
class BuyMovieTickets(ConfirmedCommand):
    movie_name: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    number_of_tickets: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    location: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    show_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    show_time: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    show_type: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
