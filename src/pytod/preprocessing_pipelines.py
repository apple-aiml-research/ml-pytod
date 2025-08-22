#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain
from typing import Any, Callable, Literal

from pytod.preprocessing_utils import ConversationPrefix
from pytod.pytod_types.aliases import IntentName, ServiceName, SlotName
from pytod.pytod_types.sgd_conversation import Conversation, Turn

logger = logging.getLogger(__name__)

PipelineStep = Callable[[Conversation, ...], ConversationPrefix]
# list containing the schema of each SGD service
SgdJSONSchema = list[dict[str, Any]]


def callable_name(any_callable: Callable[..., Any]) -> str:
    if isinstance(any_callable, partial):
        return any_callable.func.__name__

    try:
        return any_callable.__name__
    except AttributeError:
        return str(any_callable)


def modify_slot_names(
    schema: SgdJSONSchema, service: ServiceName, slot_map: dict[SlotName, str]
) -> SgdJSONSchema:
    """Modify slot names from a raw SGD schema.

    Parameters
    ---------
    schema
        A collection of all SGD train/dev/test service schemata.
    service
        Which service schema is to be modified.
    slot_map
        Mapping from the original SGD slot name to the new slot name.

    Returns
    ------
    transformed_schema
    """

    def modify_slot_schemata(
        slot_map: dict[SlotName, str], slots_schemata: list[dict[str, Any]]
    ):
        for slot_schema in slots_schemata:
            if (sgd_slot_name := slot_schema["name"]) in slot_map:
                slot_schema["name"] = slot_map[sgd_slot_name]

    def modify_required_or_result_slots(
        slot_map: dict[SlotName, str],
        intent_schemata: list[dict[str, Any]],
        key: Literal["required_slots", "result_slots"],
    ):
        for schema in intent_schemata:
            modified_slots = []
            to_modify = set(schema[key]).intersection(slot_map)
            if not to_modify:
                continue
            for req_slot in schema[key]:
                if req_slot in slot_map:
                    modified_slots.append(slot_map[req_slot])
                else:
                    modified_slots.append(req_slot)
            schema[key] = modified_slots

    def modify_optional_slots(
        slot_map: dict[SlotName, str], intent_schemata: list[dict[str, Any]]
    ):
        for schema in intent_schemata:
            to_modify = set(slot_map.keys()).intersection(schema["optional_slots"])
            while to_modify:
                modified_slot = to_modify.pop()
                opt_slot_value = schema["optional_slots"].pop(modified_slot)
                schema["optional_slots"].update(
                    {slot_map[modified_slot]: opt_slot_value}
                )

    schema = deepcopy(schema)
    transformed_schema = []
    for service_schema in schema:
        if service_schema["service_name"] == service:
            modify_slot_schemata(slot_map, service_schema["slots"])
            modify_required_or_result_slots(
                slot_map, service_schema["intents"], "required_slots"
            )
            modify_optional_slots(slot_map, service_schema["intents"])
            modify_required_or_result_slots(
                slot_map, service_schema["intents"], "result_slots"
            )
        transformed_schema.append(service_schema)

    return transformed_schema


def add_data_types(schema: SgdJSONSchema) -> list[dict[str, Any]]:
    """Add data type information to slot schema."""

    class SlotDataTypes(Enum):
        int = "int"
        str = "str"
        bool = "bool"
        float = "float"
        enum = "enum"

    def is_float(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    for service_schema in schema:
        for slot_schema in service_schema["slots"]:
            if set(slot_schema["possible_values"]) == {"True", "False"}:
                data_type = SlotDataTypes.bool
            elif slot_schema["is_categorical"]:
                if all([v.isdigit() for v in slot_schema["possible_values"]]):
                    data_type = SlotDataTypes.int
                elif all([is_float(v) for v in slot_schema["possible_values"]]):
                    data_type = SlotDataTypes.float
                else:
                    data_type = SlotDataTypes.enum
            else:
                data_type = SlotDataTypes.str
            slot_schema["data_type"] = data_type.value
    return schema


def modify_return_types(
    schema: SgdJSONSchema,
    requested_slots_metadata: dict[ServiceName, dict[str, Any]],
) -> SgdJSONSchema:
    """Modify slot names from a raw SGD schema.

    Parameters
    ---------
    schema
        A collection of all SGD train/dev/test service schemata.

    requested_slots_metadata
        For each service, we define:

           - `informable_and_requestable`: list[SlotName] communicated or requested by user
           -  `req_not_args` mapping of the form::

               {
                   'slot_name': list[IntentName] from the SGD schema
               }

               of what slots are requested during the specified intent even though they are not arg
           - `req_opt_args`: same structure as `req_not_args`. Identifies optional slots the user
           can request
           - `requestable` all slots that can be requested (or which the user can check the value or
           sys can provide values for)

    Returns
    ------
    transformed_schema
    """

    def add_return_signature(
        info_and_request_args: list[SlotName], intent_schemata: list[dict[str, Any]]
    ):
        for schema in intent_schemata:
            req_or_opt_args = set(schema["optional_slots"].keys()).union(
                schema["required_slots"]
            )
            api_returns = []
            results = schema["result_slots"]
            for slot in results:
                if slot in req_or_opt_args and slot in info_and_request_args:
                    api_returns.append(slot)
                elif slot not in req_or_opt_args:
                    api_returns.append(slot)
                else:
                    continue
            schema["api_returns"] = api_returns
            if not schema["api_returns"] == schema["result_slots"]:
                logger.info(
                    f"Transformed return type for intent: {schema['name']} \n"
                    f"Dropped slot: {list(set(results).difference(api_returns))} \n"
                    f"Proposed return type: {api_returns}\n"
                )

    schema = deepcopy(schema)
    transformed_schema = []
    for service_schema in schema:
        service = service_schema["service_name"]
        logger.info(f"Defining API return types for service {service}")
        add_return_signature(
            requested_slots_metadata[service]["informable_and_requestable"],
            service_schema["intents"],
        )
        transformed_schema.append(service_schema)

    return transformed_schema


def add_metadata_to_intent_schema(
    schema: SgdJSONSchema,
    metadata: dict[ServiceName, dict[IntentName, dict[str, Any]]],
) -> SgdJSONSchema:
    """Add additional information to the intent schema to support
    building `python` commands representing the intents directly
    from the schema.

    Parameters
    ----------
    schema
        A collection of all SGD train/dev/test service schemata.
    metadata
        Additional metadata to be added to the intent schema. See
        src/pytod/configs/pytod_setup/metadata/schema_augmentation.yaml
        for the metadata definition.
    """

    for service_schema in schema:
        service_metadata = metadata[service_schema["service_name"]]
        for intent_schema in service_schema["intents"]:
            intent_metadata = service_metadata[intent_schema["name"]]
            intent_schema.update(intent_metadata)
            if (p := intent_schema["shared_properties"]) is not None:
                assert all(isinstance(el, str) for el in p)

    return schema


def add_wildcard_info_to_slot_schema(
    schema: list[dict[str, Any]], wildcard_slots: dict[ServiceName, list[SlotName]]
) -> list[dict[str, Any]]:
    """Adds the `accepts_wildcard` attribute to the slot schema, to mark slots
    which can take the special `dontcare` value."""
    for service_schema in schema:
        service = service_schema["service_name"]
        for slot_schema in service_schema["slots"]:
            slot_schema["accepts_wildcard"] = False
            if slot_schema["name"] in wildcard_slots[service]:
                slot_schema["accepts_wildcard"] = True
    return schema


def add_followup_metadata(schema: SgdJSONSchema) -> SgdJSONSchema:
    """Add metadata defining properties shared between intents and
    follow-up intents in the same service."""

    def assert_on_system_tracked_slots_behaviour(intent_schema: dict[str, Any]):
        try:
            if intent_schema["system_tracked_slots"] is not None:
                assert not intent_schema["system_tracked_slots"].intersection(
                    intent_schema["required_slots"]
                )
        except KeyError:  # not relevant for confirmed APIs
            pass

    def get_all_followup_args(
        service_schema: dict[str, Any], follow_up: IntentName
    ) -> list[SlotName]:
        for intent_schema in service_schema["intents"]:
            if intent_schema["name"] == follow_up:
                return intent_schema["required_slots"] + list(
                    intent_schema["optional_slots"].keys()
                )
        logger.warning(f"Could not find schema for intent {follow_up}.")
        assert followup_intent in {"BuyMovieTickets"}, service_schema[
            "service_name"
        ] in {"Movies_1"}
        for intent_schema in service_schema["intents"]:
            if intent_schema["name"] == "GetTimesForMovie":
                return intent_schema["required_slots"] + list(opt_args.keys())

    for service_schema in schema:
        for intent_schema in service_schema["intents"]:
            req_args = intent_schema["required_slots"]
            opt_args = intent_schema["optional_slots"]
            try:
                sys_tracked = intent_schema["system_tracked_slots"] or []
            except KeyError:
                sys_tracked = []
                # sys_tracked = intent_schema["system_confirmed_slots"] or []
            shared_properties = intent_schema["shared_properties"] or []
            transactional = intent_schema["is_transactional"]
            try:
                followup_intent = intent_schema["followup_command"]
            except KeyError:
                raise KeyError
            if transactional or (
                not transactional and len(service_schema["intents"]) == 1
            ):
                assert followup_intent is None
                assert_on_system_tracked_slots_behaviour(intent_schema)
                if shared_properties is None:
                    intent_schema["followup_metadata"] = {
                        "required": req_args
                        + [s for s in sys_tracked if s not in req_args]
                        or None,
                    }
                    intent_schema["followup_metadata"]["optional"] = [
                        s
                        for s in list(opt_args.keys())
                        if s
                        not in (intent_schema["followup_metadata"]["required"] or [])
                    ] or None
                else:
                    intent_schema["followup_metadata"] = {
                        "required": None,
                        "optional": None,
                    }
            elif not transactional and followup_intent is None:
                intent_schema["followup_metadata"] = {
                    "required": None,
                    "optional": None,
                }
            else:
                assert followup_intent is not None
                follow_up_args = get_all_followup_args(service_schema, followup_intent)
                intent_schema["followup_metadata"] = {
                    "required": [
                        r
                        for r in req_args
                        + [s for s in sys_tracked if s not in req_args]
                        if r in follow_up_args
                    ]
                    or None,
                }
                intent_schema["followup_metadata"]["optional"] = [
                    r
                    for r in opt_args
                    if r in follow_up_args
                    and r not in intent_schema["followup_metadata"]["required"]
                ] or None
            current_meta = list(
                chain(
                    *[
                        v
                        for v in intent_schema["followup_metadata"].values()
                        if v is not None
                    ]
                )
            )
            intent_schema["followup_metadata"]["public_members"] = [
                s for s in shared_properties if s not in current_meta
            ] or None
    return schema


class PipelineMixin:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run_step(
        self, step: PipelineStep, conversation: Conversation, **kwargs
    ) -> ConversationPrefix:
        step_name = callable_name(step)
        runtime_args = kwargs.get(step_name, {})
        step_results = step(conversation, **runtime_args)

        return step_results

    def __call__(self, conversation: Conversation, **kwargs):
        pass


class PrefixExtractionPipeline(PipelineMixin):
    def __init__(self, steps: list[PipelineStep]):
        super().__init__(steps)

    def __call__(self, conversation: Conversation, **kwargs) -> ConversationPrefix:
        prefix = self.run_step(self.steps[0], conversation, **kwargs)
        last_turn_index = prefix.last_turn_idx
        for step in self.steps[1:]:
            prefix = self.run_step(step, prefix.prefix, **kwargs)
            if prefix.last_turn_idx != -1:
                if last_turn_index == -1:
                    last_turn_index = prefix.last_turn_idx
                else:
                    last_turn_index = min(prefix.last_turn_idx, last_turn_index)
        return ConversationPrefix(prefix=prefix.prefix, last_turn_idx=last_turn_index)


class FilteringPipeline(PipelineMixin):
    def __call__(self, conversation: Conversation, **kwargs) -> bool:
        for step in self.steps:
            result = self.run_step(step, conversation, **kwargs)
            if result:
                return True
        return False


class SchemaPreprocessingPipeline:
    def __init__(self, steps: list[Callable[[dict[str, Any], ...], dict[str, Any]]]):
        self.steps = steps

    @staticmethod
    def run_step(
        step: Callable[[dict[str, Any], ...], dict[str, Any]],
        schema: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        step_name = callable_name(step)
        runtime_args = kwargs.get(step_name, {})
        step_results = step(schema, **runtime_args)

        return step_results

    def __call__(self, schema: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = deepcopy(schema)
        for step in self.steps:
            result = self.run_step(step, result, **kwargs)
        return result


class ActionsPreprocessingPipeline:
    """In-place processor for the actions of a given turn."""

    def __init__(self, steps: list[Callable[[Turn, ServiceName, ...], None]]):
        self.steps = steps

    def run_step(
        self,
        step: Callable[[Turn, ServiceName, ...], None],
        turn: Turn,
        active_service: ServiceName,
        **kwargs,
    ):
        step_name = callable_name(step)
        runtime_args = kwargs.get(step_name, {})
        step(turn, active_service, **runtime_args)

    def __call__(self, turn: Turn, active_service: ServiceName, **kwargs):
        for step in self.steps:
            self.run_step(step, turn, active_service, **kwargs)


def no_op_prefix_filter(dial: Conversation) -> ConversationPrefix:
    return ConversationPrefix(prefix=dial, last_turn_idx=-1)
