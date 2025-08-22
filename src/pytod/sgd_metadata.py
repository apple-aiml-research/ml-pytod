#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional

from pydantic import BaseModel

NON_CUMULATIVE_SERVICES = {
    "Payment_1",
    "Payment_11",
    "Payment_12",
    "Payment_13",
    "Payment_14",
    "Payment_15",
}

entities = {
    "Banks_1": {"CheckBalance": [], "TransferMoney": []},
    "Buses_1": {"FindBus": ["leaving_time"], "BuyBusTicket": []},
    "Buses_2": {"BuyBusTicket": [], "FindBus": ["departure_time"]},
    "Calendar_1": {
        "GetAvailableTime": ["event_name", "event_time"],
        "AddEvent": [],
        "GetEvents": ["event_name", "event_time"],
    },
    "Events_1": {"FindEvents": ["event_name"], "BuyEventTickets": []},
    "Events_2": {
        "FindEvents": ["event_name"],
        "BuyEventTickets": [],
        "GetEventDates": ["date"],
    },
    "Flights_1": {
        "ReserveRoundtripFlights": [],
        "SearchOnewayFlight": ["inbound_departure_time", "outbound_departure_time"],
        "SearchRoundtripFlights": ["inbound_departure_time", "outbound_departure_time"],
        "ReserveOnewayFlight": ["outbound_departure_time"],
    },
    "Flights_2": {"SearchRoundtripFlights": [], "SearchOnewayFlight": []},
    "Homes_1": {"FindApartment": ["property_name"], "ScheduleVisit": []},
    "Hotels_1": {"SearchHotel": ["hotel_name"], "ReserveHotel": []},
    "Hotels_2": {"SearchHouse": [], "BookHouse": []},
    "Hotels_3": {"SearchHotel": ["hotel_name"], "ReserveHotel": []},
    "Media_1": {"FindMovies": ["title"], "PlayMovie": []},
    "Movies_1": {
        "BuyMovieTickets": [],
        "FindMovies": ["movie_name"],
        "GetTimesForMovie": ["show_time"],
    },
    "Music_1": {"LookupSong": ["song_name"], "PlaySong": []},
    "Music_2": {"LookupMusic": ["song_name"], "PlayMedia": []},
    "RentalCars_1": {"GetCarsAvailable": ["pickup_location"], "ReserveCar": []},
    "RentalCars_2": {"GetCarsAvailable": ["pickup_location"], "ReserveCar": []},
    "Restaurants_1": {"FindRestaurants": ["restaurant_name"], "ReserveRestaurant": []},
    "RideSharing_1": {"GetRide": []},
    "RideSharing_2": {"GetRide": []},
    "Services_1": {"FindProvider": ["stylist_name"], "BookAppointment": []},
    "Services_2": {"FindProvider": ["dentist_name"], "BookAppointment": []},
    "Services_3": {"FindProvider": ["doctor_name"], "BookAppointment": []},
    "Travel_1": {"FindAttractions": []},
    "Weather_1": {"GetWeather": []},
    "Alarm_1": {"GetAlarms": [], "AddAlarm": []},
    "Banks_2": {"CheckBalance": [], "TransferMoney": []},
    "Flights_3": {"SearchRoundtripFlights": [], "SearchOnewayFlight": []},
    "Hotels_4": {"SearchHotel": ["place_name"], "ReserveHotel": []},
    "Media_2": {"FindMovies": ["movie_name"], "RentMovie": []},
    "Movies_2": {"FindMovies": []},
    "Restaurants_2": {"FindRestaurants": ["restaurant_name"], "ReserveRestaurant": []},
    "Services_4": {"FindProvider": ["therapist_name"], "BookAppointment": []},
    "Buses_3": {"FindBus": ["departure_time"], "BuyBusTicket": []},
    "Events_3": {"FindEvents": ["event_name"], "BuyEventTickets": []},
    "Flights_4": {"SearchRoundtripFlights": [], "SearchOnewayFlight": []},
    "Homes_2": {"FindHomeByArea": ["property_name"], "ScheduleVisit": []},
    "Media_3": {"FindMovies": ["title"], "PlayMovie": []},
    "Messaging_1": {"ShareLocation": []},
    "Movies_3": {"FindMovies": []},
    "Music_3": {"LookupMusic": ["track"], "PlayMedia": []},
    "Payment_1": {"MakePayment": [], "RequestPayment": []},
    "RentalCars_3": {"ReserveCar": [], "GetCarsAvailable": ["pickup_location"]},
    "Trains_1": {"FindTrains": ["journey_start_time"], "GetTrainTickets": []},
}
raw_requested_metadata = {
    "Alarm_1": {
        "informable_and_requestable": [],
        "req_not_args": {},
        "req_opt_args": {},
        "requestable": [],
    },
    "Banks_2": {
        "informable_and_requestable": [],
        "req_not_args": {"transfer_time": ["TransferMoney"]},
        "req_opt_args": {},
        "requestable": ["transfer_time"],
    },
    "Buses_1": {
        "informable_and_requestable": [],
        "req_not_args": {
            "fare": ["BuyBusTicket"],
            "from_station": ["BuyBusTicket", "FindBus"],
            "to_station": ["BuyBusTicket", "FindBus"],
            "transfers": ["BuyBusTicket"],
        },
        "req_opt_args": {},
        "requestable": ["transfers", "from_station", "to_station", "fare"],
    },
    "Events_1": {
        "informable_and_requestable": ["subcategory"],
        "req_not_args": {
            "address_of_location": ["BuyEventTickets", "FindEvents"],
            "event_location": ["BuyEventTickets"],
            "time": ["BuyEventTickets"],
        },
        "req_opt_args": {"subcategory": ["FindEvents"]},
        "requestable": [
            "subcategory",
            "time",
            "event_location",
            "address_of_location",
        ],
    },
    "Flights_3": {
        "informable_and_requestable": [
            "number_checked_bags",
            "flight_class",
            "passengers",
        ],
        "req_not_args": {
            "arrives_next_day": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "destination_airport_name": [
                "SearchOnewayFlight",
                "SearchRoundtripFlights",
            ],
            "inbound_arrival_time": ["SearchRoundtripFlights"],
            "number_stops": ["SearchRoundtripFlights"],
            "origin_airport_name": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "outbound_arrival_time": ["SearchOnewayFlight", "SearchRoundtripFlights"],
        },
        "req_opt_args": {
            "flight_class": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "number_checked_bags": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "passengers": ["SearchOnewayFlight", "SearchRoundtripFlights"],
        },
        "requestable": [
            "number_checked_bags",
            "destination_airport_name",
            "inbound_arrival_time",
            "number_stops",
            "outbound_arrival_time",
            "origin_airport_name",
            "flight_class",
            "arrives_next_day",
            "passengers",
        ],
    },
    "Homes_1": {
        "informable_and_requestable": ["furnished", "pets_allowed"],
        "req_not_args": {
            "furnished": ["ScheduleVisit"],
            "pets_allowed": ["ScheduleVisit"],
            "phone_number": ["FindApartment", "ScheduleVisit"],
        },
        "req_opt_args": {
            "furnished": ["FindApartment"],
            "pets_allowed": ["FindApartment"],
        },
        "requestable": ["phone_number", "furnished", "pets_allowed"],
    },
    "Hotels_1": {
        "informable_and_requestable": ["star_rating", "destination", "has_wifi"],
        "req_not_args": {
            "has_wifi": ["ReserveHotel"],
            "phone_number": ["ReserveHotel", "SearchHotel"],
            "price_per_night": ["ReserveHotel", "SearchHotel"],
            "star_rating": ["ReserveHotel"],
            "street_address": ["ReserveHotel", "SearchHotel"],
        },
        "req_opt_args": {"has_wifi": ["SearchHotel"]},
        "requestable": [
            "phone_number",
            "price_per_night",
            "destination",
            "has_wifi",
            "street_address",
            "star_rating",
        ],
    },
    "Hotels_4": {
        "informable_and_requestable": ["smoking_allowed"],
        "req_not_args": {
            "phone_number": ["ReserveHotel", "SearchHotel"],
            "price_per_night": ["ReserveHotel", "SearchHotel"],
            "smoking_allowed": ["ReserveHotel"],
            "street_address": ["ReserveHotel", "SearchHotel"],
        },
        "req_opt_args": {"smoking_allowed": ["SearchHotel"]},
        "requestable": [
            "phone_number",
            "price_per_night",
            "street_address",
            "smoking_allowed",
        ],
    },
    "Media_2": {
        "informable_and_requestable": ["director", "actors"],
        "req_not_args": {
            "actors": ["RentMovie"],
            "director": ["RentMovie"],
            "price": ["RentMovie"],
        },
        "req_opt_args": {},
        "requestable": ["director", "actors", "price"],
    },
    "Movies_2": {
        "informable_and_requestable": ["genre", "director", "starring"],
        "req_not_args": {},
        "req_opt_args": {
            "director": ["FindMovies"],
            "genre": ["FindMovies"],
            "starring": ["FindMovies"],
        },
        "requestable": ["genre", "director", "starring"],
    },
    "Music_1": {
        "informable_and_requestable": ["artist", "year", "album", "genre"],
        "req_not_args": {
            "album": ["PlaySong"],
            "genre": ["PlaySong"],
            "year": ["PlaySong"],
        },
        "req_opt_args": {
            "artist": ["PlaySong"],
            "genre": ["LookupSong"],
            "year": ["LookupSong"],
        },
        "requestable": ["artist", "genre", "album", "year"],
    },
    "RentalCars_1": {
        "informable_and_requestable": [],
        "req_not_args": {
            "car_name": ["ReserveCar"],
            "total_price": ["GetCarsAvailable", "ReserveCar"],
        },
        "req_opt_args": {},
        "requestable": ["total_price", "car_name"],
    },
    "Restaurants_2": {
        "informable_and_requestable": [
            "has_vegetarian_options",
            "price_range",
            "category",
            "has_seating_outdoors",
        ],
        "req_not_args": {
            "address": ["FindRestaurants", "ReserveRestaurant"],
            "category": ["ReserveRestaurant"],
            "has_seating_outdoors": ["ReserveRestaurant"],
            "has_vegetarian_options": ["ReserveRestaurant"],
            "phone_number": ["FindRestaurants", "ReserveRestaurant"],
            "price_range": ["ReserveRestaurant"],
            "rating": ["FindRestaurants", "ReserveRestaurant"],
        },
        "req_opt_args": {
            "has_seating_outdoors": ["FindRestaurants"],
            "has_vegetarian_options": ["FindRestaurants"],
            "price_range": ["FindRestaurants"],
        },
        "requestable": [
            "category",
            "address",
            "rating",
            "has_vegetarian_options",
            "phone_number",
            "has_seating_outdoors",
            "price_range",
        ],
    },
    "RideSharing_1": {
        "informable_and_requestable": [],
        "req_not_args": {
            "approximate_ride_duration": ["GetRide"],
            "ride_fare": ["GetRide"],
        },
        "req_opt_args": {},
        "requestable": ["ride_fare", "approximate_ride_duration"],
    },
    "Services_4": {
        "informable_and_requestable": [],
        "req_not_args": {
            "address": ["BookAppointment", "FindProvider"],
            "phone_number": ["BookAppointment", "FindProvider"],
        },
        "req_opt_args": {},
        "requestable": ["phone_number", "address"],
    },
    "Travel_1": {
        "informable_and_requestable": ["free_entry", "good_for_kids"],
        "req_not_args": {"phone_number": ["FindAttractions"]},
        "req_opt_args": {
            "free_entry": ["FindAttractions"],
            "good_for_kids": ["FindAttractions"],
        },
        "requestable": ["phone_number", "free_entry", "good_for_kids"],
    },
    "Weather_1": {
        "informable_and_requestable": ["date"],
        "req_not_args": {"humidity": ["GetWeather"], "wind": ["GetWeather"]},
        "req_opt_args": {"date": ["GetWeather"]},
        "requestable": ["humidity", "wind", "date"],
    },
    "Buses_3": {
        "informable_and_requestable": ["category"],
        "req_not_args": {
            "category": ["BuyBusTicket"],
            "from_station": ["BuyBusTicket", "FindBus"],
            "price": ["BuyBusTicket"],
            "to_station": ["BuyBusTicket", "FindBus"],
        },
        "req_opt_args": {"category": ["FindBus"]},
        "requestable": ["from_station", "to_station", "category", "price"],
    },
    "Events_3": {
        "informable_and_requestable": [],
        "req_not_args": {
            "price_per_ticket": ["BuyEventTickets", "FindEvents"],
            "time": ["BuyEventTickets"],
            "venue": ["BuyEventTickets"],
            "venue_address": ["BuyEventTickets", "FindEvents"],
        },
        "req_opt_args": {},
        "requestable": ["time", "venue", "venue_address", "price_per_ticket"],
    },
    "Flights_4": {
        "informable_and_requestable": ["number_of_tickets", "seating_class"],
        "req_not_args": {
            "inbound_arrival_time": ["SearchRoundtripFlights"],
            "outbound_arrival_time": ["SearchOnewayFlight", "SearchRoundtripFlights"],
        },
        "req_opt_args": {
            "number_of_tickets": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "seating_class": ["SearchOnewayFlight", "SearchRoundtripFlights"],
        },
        "requestable": [
            "number_of_tickets",
            "seating_class",
            "inbound_arrival_time",
            "outbound_arrival_time",
        ],
    },
    "Homes_2": {
        "informable_and_requestable": [
            "number_of_beds",
            "number_of_baths",
            "in_unit_laundry",
            "has_garage",
        ],
        "req_not_args": {
            "address": ["ScheduleVisit"],
            "has_garage": ["ScheduleVisit"],
            "in_unit_laundry": ["ScheduleVisit"],
            "number_of_baths": ["ScheduleVisit"],
            "number_of_beds": ["ScheduleVisit"],
            "phone_number": ["FindHomeByArea", "ScheduleVisit"],
            "price": ["ScheduleVisit"],
        },
        "req_opt_args": {
            "has_garage": ["FindHomeByArea"],
            "in_unit_laundry": ["FindHomeByArea"],
        },
        "requestable": [
            "phone_number",
            "number_of_baths",
            "has_garage",
            "number_of_beds",
            "address",
            "price",
            "in_unit_laundry",
        ],
    },
    "Hotels_2": {
        "informable_and_requestable": ["has_laundry_service", "rating"],
        "req_not_args": {
            "address": ["BookHouse"],
            "has_laundry_service": ["BookHouse"],
            "phone_number": ["BookHouse", "SearchHouse"],
            "rating": ["BookHouse"],
            "total_price": ["BookHouse", "SearchHouse"],
        },
        "req_opt_args": {"has_laundry_service": ["SearchHouse"]},
        "requestable": [
            "phone_number",
            "total_price",
            "address",
            "has_laundry_service",
            "rating",
        ],
    },
    "Media_3": {
        "informable_and_requestable": ["genre", "starring"],
        "req_not_args": {"genre": ["PlayMovie"], "starring": ["PlayMovie"]},
        "req_opt_args": {},
        "requestable": ["genre", "starring"],
    },
    "Messaging_1": {
        "informable_and_requestable": [],
        "req_not_args": {},
        "req_opt_args": {},
        "requestable": [],
    },
    "Movies_1": {
        "informable_and_requestable": ["genre"],
        "req_not_args": {
            "genre": ["GetTimesForMovie"],
            "price": ["GetTimesForMovie"],
            "street_address": ["GetTimesForMovie"],
        },
        "req_opt_args": {},
        "requestable": ["genre", "street_address", "price"],
    },
    "Movies_3": {
        "informable_and_requestable": ["genre", "directed_by", "cast"],
        "req_not_args": {},
        "req_opt_args": {
            "cast": ["FindMovies"],
            "directed_by": ["FindMovies"],
            "genre": ["FindMovies"],
        },
        "requestable": ["genre", "directed_by", "cast"],
    },
    "Music_3": {
        "informable_and_requestable": ["genre", "year"],
        "req_not_args": {"genre": ["PlayMedia"], "year": ["PlayMedia"]},
        "req_opt_args": {"genre": ["LookupMusic"], "year": ["LookupMusic"]},
        "requestable": ["genre", "year"],
    },
    "Payment_1": {
        "informable_and_requestable": [],
        "req_not_args": {},
        "req_opt_args": {},
        "requestable": [],
    },
    "RentalCars_3": {
        "informable_and_requestable": [],
        "req_not_args": {
            "car_name": ["ReserveCar"],
            "price_per_day": ["GetCarsAvailable", "ReserveCar"],
        },
        "req_opt_args": {},
        "requestable": ["price_per_day", "car_name"],
    },
    "RideSharing_2": {
        "informable_and_requestable": [],
        "req_not_args": {"ride_fare": ["GetRide"], "wait_time": ["GetRide"]},
        "req_opt_args": {},
        "requestable": ["wait_time", "ride_fare"],
    },
    "Services_1": {
        "informable_and_requestable": ["is_unisex", "city"],
        "req_not_args": {
            "average_rating": ["BookAppointment", "FindProvider"],
            "city": ["BookAppointment"],
            "is_unisex": ["BookAppointment"],
            "phone_number": ["BookAppointment", "FindProvider"],
            "street_address": ["BookAppointment", "FindProvider"],
        },
        "req_opt_args": {"is_unisex": ["FindProvider"]},
        "requestable": [
            "average_rating",
            "phone_number",
            "street_address",
            "is_unisex",
            "city",
        ],
    },
    "Trains_1": {
        "informable_and_requestable": [],
        "req_not_args": {
            "from_station": ["FindTrains", "GetTrainTickets"],
            "to_station": ["FindTrains", "GetTrainTickets"],
        },
        "req_opt_args": {},
        "requestable": ["from_station", "to_station"],
    },
    "Banks_1": {
        "informable_and_requestable": [],
        "req_not_args": {},
        "req_opt_args": {},
        "requestable": [],
    },
    "Buses_2": {
        "informable_and_requestable": [],
        "req_not_args": {
            "destination_station_name": ["BuyBusTicket", "FindBus"],
            "origin_station_name": ["BuyBusTicket", "FindBus"],
            "price": ["BuyBusTicket"],
        },
        "req_opt_args": {},
        "requestable": [
            "destination_station_name",
            "price",
            "origin_station_name",
        ],
    },
    "Calendar_1": {
        "informable_and_requestable": [],
        "req_not_args": {},
        "req_opt_args": {},
        "requestable": [],
    },
    "Events_2": {
        "informable_and_requestable": ["category"],
        "req_not_args": {
            "time": ["BuyEventTickets", "FindEvents", "GetEventDates"],
            "venue": ["BuyEventTickets"],
            "venue_address": ["BuyEventTickets", "FindEvents", "GetEventDates"],
        },
        "req_opt_args": {"category": ["FindEvents"]},
        "requestable": ["time", "venue", "category", "venue_address"],
    },
    "Flights_1": {
        "informable_and_requestable": ["outbound_departure_time", "refundable"],
        "req_not_args": {
            "destination_airport": [
                "ReserveOnewayFlight",
                "ReserveRoundtripFlights",
                "SearchOnewayFlight",
                "SearchRoundtripFlights",
            ],
            "inbound_arrival_time": [
                "ReserveRoundtripFlights",
                "SearchRoundtripFlights",
            ],
            "number_stops": [
                "ReserveOnewayFlight",
                "ReserveRoundtripFlights",
                "SearchRoundtripFlights",
            ],
            "origin_airport": [
                "ReserveOnewayFlight",
                "ReserveRoundtripFlights",
                "SearchOnewayFlight",
                "SearchRoundtripFlights",
            ],
            "outbound_arrival_time": [
                "ReserveOnewayFlight",
                "ReserveRoundtripFlights",
                "SearchOnewayFlight",
                "SearchRoundtripFlights",
            ],
            "outbound_departure_time": ["ReserveOnewayFlight"],
            "price": ["ReserveOnewayFlight", "ReserveRoundtripFlights"],
        },
        "req_opt_args": {
            "refundable": [
                "ReserveOnewayFlight",
                "ReserveRoundtripFlights",
                "SearchOnewayFlight",
                "SearchRoundtripFlights",
            ]
        },
        "requestable": [
            "refundable",
            "destination_airport",
            "inbound_arrival_time",
            "number_stops",
            "outbound_arrival_time",
            "outbound_departure_time",
            "price",
            "origin_airport",
        ],
    },
    "Flights_2": {
        "informable_and_requestable": ["passengers", "seating_class"],
        "req_not_args": {
            "destination_airport": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "inbound_arrival_time": ["SearchRoundtripFlights"],
            "is_redeye": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "number_stops": ["SearchRoundtripFlights"],
            "origin_airport": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "outbound_arrival_time": ["SearchOnewayFlight", "SearchRoundtripFlights"],
        },
        "req_opt_args": {
            "passengers": ["SearchOnewayFlight", "SearchRoundtripFlights"],
            "seating_class": ["SearchOnewayFlight", "SearchRoundtripFlights"],
        },
        "requestable": [
            "is_redeye",
            "seating_class",
            "destination_airport",
            "inbound_arrival_time",
            "number_stops",
            "outbound_arrival_time",
            "passengers",
            "origin_airport",
        ],
    },
    "Hotels_3": {
        "informable_and_requestable": ["pets_welcome", "location"],
        "req_not_args": {
            "average_rating": ["ReserveHotel"],
            "pets_welcome": ["ReserveHotel"],
            "phone_number": ["ReserveHotel", "SearchHotel"],
            "price": ["ReserveHotel", "SearchHotel"],
            "street_address": ["ReserveHotel", "SearchHotel"],
        },
        "req_opt_args": {"pets_welcome": ["SearchHotel"]},
        "requestable": [
            "average_rating",
            "pets_welcome",
            "phone_number",
            "street_address",
            "location",
            "price",
        ],
    },
    "Media_1": {
        "informable_and_requestable": ["genre", "directed_by"],
        "req_not_args": {"directed_by": ["PlayMovie"], "genre": ["PlayMovie"]},
        "req_opt_args": {},
        "requestable": ["genre", "directed_by"],
    },
    "Music_2": {
        "informable_and_requestable": ["artist", "album", "genre"],
        "req_not_args": {"album": ["PlayMedia"], "genre": ["PlayMedia"]},
        "req_opt_args": {"artist": ["PlayMedia"], "genre": ["LookupMusic"]},
        "requestable": ["artist", "album", "genre"],
    },
    "RentalCars_2": {
        "informable_and_requestable": [],
        "req_not_args": {
            "car_name": ["ReserveCar"],
            "total_price": ["GetCarsAvailable", "ReserveCar"],
        },
        "req_opt_args": {},
        "requestable": ["total_price", "car_name"],
    },
    "Restaurants_1": {
        "informable_and_requestable": [
            "has_live_music",
            "cuisine",
            "price_range",
            "serves_alcohol",
        ],
        "req_not_args": {
            "cuisine": ["ReserveRestaurant"],
            "has_live_music": ["ReserveRestaurant"],
            "phone_number": ["FindRestaurants", "ReserveRestaurant"],
            "price_range": ["ReserveRestaurant"],
            "serves_alcohol": ["ReserveRestaurant"],
            "street_address": ["FindRestaurants", "ReserveRestaurant"],
        },
        "req_opt_args": {
            "has_live_music": ["FindRestaurants"],
            "price_range": ["FindRestaurants"],
            "serves_alcohol": ["FindRestaurants"],
        },
        "requestable": [
            "phone_number",
            "serves_alcohol",
            "has_live_music",
            "street_address",
            "cuisine",
            "price_range",
        ],
    },
    "Services_2": {
        "informable_and_requestable": ["city", "offers_cosmetic_services"],
        "req_not_args": {
            "address": ["BookAppointment", "FindProvider"],
            "city": ["BookAppointment"],
            "offers_cosmetic_services": ["BookAppointment"],
            "phone_number": ["BookAppointment", "FindProvider"],
        },
        "req_opt_args": {"offers_cosmetic_services": ["FindProvider"]},
        "requestable": [
            "phone_number",
            "offers_cosmetic_services",
            "address",
            "city",
        ],
    },
    "Services_3": {
        "informable_and_requestable": ["type"],
        "req_not_args": {
            "average_rating": ["BookAppointment", "FindProvider"],
            "phone_number": ["BookAppointment", "FindProvider"],
            "street_address": ["BookAppointment", "FindProvider"],
            "type": ["BookAppointment"],
        },
        "req_opt_args": {},
        "requestable": [
            "phone_number",
            "average_rating",
            "street_address",
            "type",
        ],
    },
}


result_slots_raw = {
    "Banks_1": ["balance"],
    "Banks_2": ["account_balance"],
    "Alarm_1": ["alarm_name", "alarm_time"],
    "Flights_2": ["fare"],
    "Flights_3": ["price"],
    "Homes_1": ["address", "rent"],
    "Movies_2": ["title", "aggregate_rating"],
    "Trains_1": ["total"],
    "Travel_1": ["attraction_name"],
    "Weather_1": ["precipitation", "temperature"],
}


non_tracked_sys_optionals = {
    "Hotels_4": ["star_rating"],  # dev/20_00101 vs dev/20_00102
    "Travel_1": ["category"],  # dev/20_00100 vs dev/20_00102
    "Homes_1": ["number_of_baths"],  # dev/3_00112 vs dev/3_00114
}


class ServiceRequestablesMetadata(BaseModel):
    informable_and_requestable: list[str]
    requestable: list[str]
    req_opt_args: dict[str, list[str]]
    req_not_args: dict[str, list[str]]


class RequestedSlotsMetadata(BaseModel):
    info: dict[str, ServiceRequestablesMetadata]

    def is_only_requested(self, slot: str, service: str) -> bool:
        return (
            self.can_request(slot, service)
            and slot not in self.info[service].informable_and_requestable
        )

    def can_request(self, slot: str, service: str) -> bool:
        return slot in self.info[service].requestable


class ResultSlots(BaseModel):
    info: dict[str, list[str]]

    def is_result(self, slot: str, service: str) -> Optional[bool]:
        try:
            return slot in self.info[service]
        except KeyError:
            return


class NotTrackedOptionals(BaseModel):
    info: dict[str, list[str]]

    def is_untracked(self, slot: str, service: str) -> bool:
        try:
            return slot in self.info[service]
        except KeyError:
            return False


requested_slots = RequestedSlotsMetadata.model_validate(
    {"info": raw_requested_metadata}
)
result_slots = ResultSlots.model_validate({"info": result_slots_raw})
untracked_optionals = NotTrackedOptionals.model_validate(
    {"info": non_tracked_sys_optionals}
)
multivalue_offered_action_slots = {
    "Movies_1": ["movie_name"],
    "Media_1": ["title"],
    "Media_2": ["movie_name"],
    "Media_3": ["title"],
}
"""Slots where the system OFFERs multiple values at once"""

UNANNOTATED_INT_SLOTS = {"stay_length", "number_of_days"}
"""Slots which can take only integer values but are
not annotated accordingly in the schema"""
