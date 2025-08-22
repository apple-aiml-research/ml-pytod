#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging

logger = logging.getLogger(__name__)


class ValueProcessor:
    """A class that contains metadata for parsing the
    states correctly."""

    LOWER_TO_SCHEMA_CASE_MAPPING = {
        "true": "True",
        "false": "False",
        "economy": "Economy",
        "economy extra": "Economy extra",
        "flexible": "Flexible",
        "music": "Music",
        "sports": "Sports",
        "premium economy": "Premium Economy",
        "business": "Business",
        "first class": "First Class",
        "united airlines": "United Airlines",
        "american airlines": "American Airlines",
        "delta airlines": "Delta Airlines",
        "southwest airlines": "Southwest Airlines",
        "alaska airlines": "Alaska Airlines",
        "british airlines": "British Airways",
        "british airways": "British Airways",
        "air canada": "Air Canada",
        "air france": "Air France",
        "tv": {
            "Music_1": "TV",
            "Music_11": "TV",
            "Music_12": "TV",
            "Music_13": "TV",
            "Music_14": "TV",
            "Music_15": "TV",
            "Music_2": "TV",
            "Music_21": "TV",
            "Music_22": "TV",
            "Music_23": "TV",
            "Music_24": "TV",
            "Music_25": "TV",
        },
        "kitchen speaker": {
            "Music_1": "Kitchen speaker",
            "Music_11": "Kitchen speaker",
            "Music_12": "Kitchen speaker",
            "Music_13": "Kitchen speaker",
            "Music_14": "Kitchen speaker",
            "Music_15": "Kitchen speaker",
            "Music_2": "kitchen speaker",
            "Music_21": "kitchen speaker",
            "Music_22": "kitchen speaker",
            "Music_23": "kitchen speaker",
            "Music_24": "kitchen speaker",
            "Music_25": "kitchen speaker",
        },
        "bedroom speaker": {
            "Music_1": "Bedroom speaker",
            "Music_11": "Bedroom speaker",
            "Music_12": "Bedroom speaker",
            "Music_13": "Bedroom speaker",
            "Music_14": "Bedroom speaker",
            "Music_15": "Bedroom speaker",
            "Music_2": "bedroom speaker",
            "Music_21": "bedroom speaker",
            "Music_22": "bedroom speaker",
            "Music_23": "bedroom speaker",
            "Music_24": "bedroom speaker",
            "Music_25": "bedroom speaker",
        },
        "compact": "Compact",
        "standard": "Standard",
        "full-size": "Full-size",
        "pool": "Pool",
        "regular": {
            "RideSharing_1": "Regular",
            "RideSharing_11": "Regular",
            "RideSharing_12": "Regular",
            "RideSharing_13": "Regular",
            "RideSharing_14": "Regular",
            "RideSharing_15": "Regular",
            "RideSharing_2": "Regular",
            "RideSharing_21": "Regular",
            "RideSharing_22": "Regular",
            "RideSharing_23": "Regular",
            "RideSharing_24": "Regular",
            "RideSharing_25": "Regular",
            "Movies_1": "regular",
            "Movies_11": "regular",
            "Movies_12": "regular",
            "Movies_13": "regular",
            "Movies_14": "regular",
            "Movies_15": "regular",
        },
        "luxury": "Luxury",
        "gynecologist": "Gynecologist",
        "ent specialist": "ENT Specialist",
        "ophthalmologist": "Ophthalmologist",
        "general practitioner": "General Practitioner",
        "dermatologist": "Dermatologist",
        "place of worship": "Place of Worship",
        "theme park": "Theme Park",
        "museum": "Museum",
        "historical landmark": "Historical Landmark",
        "park": "Park",
        "tourist attraction": "Tourist Attraction",
        "sports venue": "Sports Venue",
        "shopping area": "Shopping Area",
        "performing arts venue": "Performing Arts Venue",
        "nature preserve": "Nature Preserve",
        "none": "None",
        "english": "English",
        "mandarin": "Mandarin",
        "spanish": "Spanish",
        "psychologist": "Psychologist",
        "family counselor": "Family Counselor",
        "psychiatrist": "Psychiatrist",
        "theater": "Theater",
        "theatre": "Theater",
        "south african airways": "South African Airways",
        "lot polish airlines": "LOT Polish Airlines",
        "latam brasil": "LATAM Brasil",
        "hindi": "Hindi",
        "french": "French",
        "living room": "Living room",
        "kitchen": "Kitchen",
        "patio": "Patio",
        "hatchback": "Hatchback",
        "sedan": "Sedan",
        "suv": "SUV",
        "value": "Value",
        # for completeness, we also store the values of lowercase
        # categorical slots here
        # category (Buses_3)
        "direct": "direct",
        "one-stop": "one-stop",
        # intent (Homes_2)
        "rent": "rent",
        "buy": "buy",
        # show_type (Movies_1)
        "3d": "3d",
        "imax": "imax",
        # payment_method (Payment_1)
        "app balance": "app balance",
        "debit card": "debit card",
        "credit card": "credit card",
        # price_range (Restaurants_2)
        "cheap": "cheap",
        "moderate": "moderate",
        "pricey": "pricey",
        "ultra high-end": "ultra high-end",
        # account_type, recipient_account_type (Banks_2)
        "checking": "checking",
        "savings": "savings",
    }
    """"Lookup table to restore categorical values to schema casing if the
    model predicts lowercases. Values my be Union[str, dict[str, str], with
    mappings when a value has different casings in different services."""

    WILDCARD_VALUE = {"dontcare"}
    """A value indicating that the user does not have a preference for a
    certain entity attribute during search. For transactions, this value
    essentially means that the user accepts the default parameters (which
    are always later confirmed by the agent)."""

    DIGIT_NORMALISATION_MAP = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "Thirteen": "13",
        "Twelve": "12",
        "Eleven": "11",
        "Ten": "10",
        "Nine": "9",
        "Eight": "8",
        "Seven": "7",
        "Six": "6",
        "Five": "5",
        "Four": "4",
        "Three": "3",
        "Two": "2",
        "One": "1",
        "Zero": "0",
    }

    @classmethod
    def restore_case(
        cls, value: str, service: str, restore_categorical_case: bool = True
    ) -> str:
        """Restore the case of a given categorical slot `value` to the original
        schema casing to ensure scoring is correct."""
        if (
            not restore_categorical_case
            or value not in cls.LOWER_TO_SCHEMA_CASE_MAPPING
        ):
            return value
        if value in cls.LOWER_TO_SCHEMA_CASE_MAPPING:
            recased_data = cls.LOWER_TO_SCHEMA_CASE_MAPPING[value]
            if isinstance(recased_data, str):
                return recased_data
            else:
                assert isinstance(recased_data, dict)
                try:
                    return recased_data[service]
                # raised if a value pred. in a different service
                # is in the re-casing table
                except KeyError:
                    return value

    @classmethod
    def map_to_digit(cls, value: str | None, map_to_digits: bool = True) -> str | None:
        if not map_to_digits or value is None:
            return value
        if value in cls.DIGIT_NORMALISATION_MAP:
            normalised = cls.DIGIT_NORMALISATION_MAP[value]
            logger.info(f"Normalised '{value}' to '{normalised}'")
            return normalised
        # for integers that are annotated as open values, the model
        # may pick the unit as well (eg 6 days) etc.
        words = value.split(" ")
        for w in words:
            if w in cls.DIGIT_NORMALISATION_MAP:
                logger.info(f"Normalised '{value}' to '{w}'")
                return w
            elif w in cls.DIGIT_NORMALISATION_MAP.values():
                for k, v in cls.DIGIT_NORMALISATION_MAP.items():
                    if v == w:
                        logger.info(f"Normalised '{value}' to '{k}'")
                        return k


def is_int(value: str) -> bool:
    try:
        _ = int(value.strip("'\""))
        _is_int = True
    except ValueError:
        digits = ValueProcessor.DIGIT_NORMALISATION_MAP
        _is_int = (
            value.strip("'\"") in digits
            or value.strip("'\"") in digits.values()
            or any(
                w in digits or w in digits.values()
                for w in value.strip("'\"").split(" ")
            )
        )
    return _is_int


def is_date(value: str) -> bool:
    weekdays = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    relative_expressions = ("today", "tomorrow", "month")
    if (
        "march" in value
        or any(v in value for v in weekdays)
        or any(v in value for v in relative_expressions)
    ):
        return True
    return False
