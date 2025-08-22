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


@register_command(service="Homes_2")
class FindHomeByArea(SearchCommand):
    area: SearchCommandArgument[str] = SearchCommandArgument()
    number_of_beds: SearchCommandArgument[str] = SearchCommandArgument()
    number_of_baths: SearchCommandArgument[str] = SearchCommandArgument()
    intent: SearchCommandArgument[str] = SearchCommandArgument()
    has_garage: SearchCommandArgument[str] = SearchCommandArgument()
    in_unit_laundry: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Homes_2")
class ScheduleVisit(ConfirmedCommand):
    property_name: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()
    visit_date: ConfirmedCommandArgument[str] = ConfirmedCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
