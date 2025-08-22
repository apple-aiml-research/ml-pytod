#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.pytod_types.aliases import DialogueID
from pytod.simulation.command_registry import register_command
from pytod.simulation.search_command import SearchCommand, SearchCommandArgument


@register_command(service="Travel_1")
class FindAttractions(SearchCommand):
    location: SearchCommandArgument[str] = SearchCommandArgument()
    category: SearchCommandArgument[str] = SearchCommandArgument()
    good_for_kids: SearchCommandArgument[str] = SearchCommandArgument()
    free_entry: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
