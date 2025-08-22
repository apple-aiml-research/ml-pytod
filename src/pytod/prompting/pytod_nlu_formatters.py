#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import random
from typing import Callable, Literal, NamedTuple

from jinja2 import Environment, StrictUndefined

from pytod.command import ArgumentDefinition, CommandCollection
from pytod.prompting.template_factories import nlu_example_template
from pytod.prompting.text2text_example_parsers import NLUConversationExample
from pytod.prompting.utils import TemplateMixin
from pytod.pytod_types.aliases import ServiceName, SlotName, SlotValue
from pytod.pytod_types.sgd_conversation import Author, Turn

# the index of a slot in the model target
TargetIndex = str
UNK_VALUE = "unanswerable"

SEMANTICALLY_SIMILAR = {
    "Buses_1": {
        "from_location": "to_location",
        "to_location": "from_location",
    },
    "Buses_2": {"origin": "destination", "destination": "origin"},
    "Events_1": {"category": "subcategory", "subcategory": "category"},
    "Events_2": {"event_type": "category", "category": "event_type"},
    "Flights_1": {
        "origin_city": "destination_city",
        "destination_city": "origin_city",
        "return_date": "departure_date",
        "departure_date": "return_date",
    },
    "Flights_2": {
        "origin": "destination",
        "destination": "origin",
        "return_date": "departure_date",
        "departure_date": "return_date",
    },
    "Homes_1": {
        "number_of_beds": "number_of_baths",
        "number_of_baths": "number_of_beds",
    },
    "Hotels_2": {"check_in_date": "check_out_date", "check_out_date": "check_in_date"},
    "Hotels_3": {"check_in_date": "check_out_date", "check_out_date": "check_in_date"},
    "Movies_1": {
        "show_type": "genre",
        "genre": "show_type",
        "location": "theater_name",
        "theater_name": "location",
    },
    "RentalCars_1": {
        "pickup_date": "dropoff_date",
        "dropoff_date": "pickup_date",
    },
    "RentalCars_2": {
        "pickup_date": "dropoff_date",
        "dropoff_date": "pickup_date",
    },
}


class ConversationFormatter:
    """Formats the dialogue history into
    alternating user and system turns."""

    @staticmethod
    def format(turns: list[Turn]) -> str:
        source = []
        for turn in turns:
            prefix = "user: " if turn.author == Author.USER else "agent: "
            source.append(f"{prefix}{turn.text}\n")
        return "".join(source).lower()


class QuestionFormatter:
    @staticmethod
    def format(slot_schema: ArgumentDefinition) -> str:
        description = f"{slot_schema.description}?"
        if slot_schema.is_categorical:
            possible_vals = slot_schema.possible_values
            # we do not list integers for integer slots
            # we will cast when setting the properties on the app
            try:
                _ = int(possible_vals[0])
            except ValueError:
                start = " Answer should be one of the following: "
                val_options = ", ".join([f'"{val}"' for val in possible_vals])
                val_options += f' or "{UNK_VALUE}"'
                description = description + start + val_options
        return description.lower()


class FormattedNLUExample(NamedTuple):
    prompt: str
    target: str
    parser_map: dict[TargetIndex, SlotName]


class NLUPromptFormatter(TemplateMixin):
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = nlu_example_template,
    ):
        super().__init__(schema, template_factory=template_factory)
        environment = Environment()
        environment.filters["conversation_formatter"] = ConversationFormatter.format
        environment.filters["question_formatter"] = QuestionFormatter.format
        self._set_template_and_variables(environment)

    def get_prompt(self, request: dict[Literal["slot_list", "source_turns"]]) -> str:
        template_vars = self._get_template_variables(request)
        return self._template.render(**template_vars, undefined=StrictUndefined)


class NLUTargetFormatter:
    @staticmethod
    def format(
        slot_list: list[ArgumentDefinition],
        targets: dict[ArgumentDefinition, SlotValue],
    ) -> str:
        target_vals = [targets.get(s, UNK_VALUE) for s in slot_list]
        return " ".join([f"{i + 1}) {t}" for i, t in enumerate(target_vals)]).lower()


class NLUExampleFormatter:
    def __init__(
        self,
        prompt_fmt: NLUPromptFormatter,
        target_fmt: NLUTargetFormatter,
        randomise_prompt_elements: bool = True,
        unk_slot_probability: float = 0.5,
        similar_slot_probability: float = 0.5,
        include_all_unk_slots: bool = False,
    ):
        self.prompt_fmt = prompt_fmt
        self.target_fmt = target_fmt
        self._randomise_prompt_elements = randomise_prompt_elements
        self._sample_unk_probability = unk_slot_probability
        self._sample_similar_slot_probability = similar_slot_probability
        self._include_all_unk_slots = include_all_unk_slots

    def format(self, example: NLUConversationExample) -> FormattedNLUExample:
        slot_list = [s for s in example.target]
        if self._include_all_unk_slots:
            slot_list += example.unknown_val_slots
        else:
            slot_list += self._maybe_sample_unk_slots(
                example.unknown_val_slots, example.service
            )
        if self._randomise_prompt_elements:
            random.shuffle(slot_list)
        prompt = self.prompt_fmt.get_prompt(
            {
                "source_turns": example.source_turns,
                "slot_list": slot_list,
            }
        )
        target = self.target_fmt.format(slot_list, example.target)
        parser_map = {f"{i + 1}": s.name for i, s in enumerate(slot_list)}
        return FormattedNLUExample(prompt=prompt, target=target, parser_map=parser_map)

    def _maybe_sample_unk_slots(
        self, unknown_val_slots: list[ArgumentDefinition], service: ServiceName
    ) -> list[ArgumentDefinition]:
        """Sample randomly a variable number of slots that have not yet been mentioned
        in the conversation."""
        if not unknown_val_slots:
            return []
        if random.random() > self._sample_unk_probability:
            return []
        try:
            n_slots = random.sample(range(1, len(unknown_val_slots)), 1)[0]
        except ValueError:
            n_slots = 1
        if (
            random.random() > self._sample_similar_slot_probability
            or service not in SEMANTICALLY_SIMILAR
        ):
            semantically_similar = []
        else:
            all_slots = [
                s for s in unknown_val_slots if s.name in SEMANTICALLY_SIMILAR[service]
            ]
            if not all_slots:
                semantically_similar = []
            elif len(all_slots) == 1 or n_slots == 1:
                semantically_similar = random.sample(all_slots, 1)
            else:
                n_similar = random.sample(range(1, n_slots), 1)[0]
                semantically_similar = random.sample(
                    all_slots, min(len(all_slots), n_similar)
                )
            n_slots -= len(semantically_similar)
        unk_slots = semantically_similar + random.sample(unknown_val_slots, n_slots)
        random.shuffle(unk_slots)
        return unk_slots
