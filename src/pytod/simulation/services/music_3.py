#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.pytod_types.aliases import DialogueID
from pytod.simulation.command_registry import register_command
from pytod.simulation.confirmed_command import (
    ConfirmedCommand,
    ConfirmedCommandArgument,
)
from pytod.simulation.search_command import SearchCommand, SearchCommandArgument


@register_command(service="Music_3")
class LookupMusic(SearchCommand):
    artist: SearchCommandArgument[str] = SearchCommandArgument()
    album: SearchCommandArgument[str] = SearchCommandArgument()
    genre: SearchCommandArgument[str] = SearchCommandArgument()
    year: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Music_3")
class PlayMedia(ConfirmedCommand):
    track: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    artist: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    device: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    album: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
