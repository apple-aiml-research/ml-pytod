#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from pytod.pytod_types.aliases import DialogueID
from pytod.simulation.command_registry import register_command
from pytod.simulation.search_command import SearchCommand, SearchCommandArgument


@register_command(service="Flights_3")
class SearchOnewayFlight(SearchCommand):
    origin_city: SearchCommandArgument[str] = SearchCommandArgument()
    destination_city: SearchCommandArgument[str] = SearchCommandArgument()
    departure_date: SearchCommandArgument[str] = SearchCommandArgument()
    airlines: SearchCommandArgument[str] = SearchCommandArgument()
    passengers: SearchCommandArgument[str] = SearchCommandArgument()
    flight_class: SearchCommandArgument[str] = SearchCommandArgument()
    number_checked_bags: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)


@register_command(service="Flights_3")
class SearchRoundtripFlights(SearchCommand):
    origin_city: SearchCommandArgument[str] = SearchCommandArgument()
    destination_city: SearchCommandArgument[str] = SearchCommandArgument()
    departure_date: SearchCommandArgument[str] = SearchCommandArgument()
    return_date: SearchCommandArgument[str] = SearchCommandArgument()
    airlines: SearchCommandArgument[str] = SearchCommandArgument()
    passengers: SearchCommandArgument[str] = SearchCommandArgument()
    flight_class: SearchCommandArgument[str] = SearchCommandArgument()
    number_checked_bags: SearchCommandArgument[str] = SearchCommandArgument()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
