#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, Literal, NamedTuple

from omegaconf import DictConfig

from pytod.command import CommandCollection
from pytod.interpreter.metadata import SKIP_QUERY_BINDING, WILDCARD_VALUE
from pytod.iterators import IndexMetadata
from pytod.preprocessing_utils import ConversationPrefix
from pytod.prompting.pytod_text2text_formatters_utils import VariableMapper
from pytod.pytod_types.aliases import (
    DialogueID,
    IntentName,
    ServiceName,
    ShardName,
    SlotName,
)
from pytod.pytod_types.pytod import PyTODConversation, SystemTurn
from pytod.pytod_types.sgd_conversation import (
    Author,
    Conversation,
    SystemDialogueAct,
    Turn,
    UserDialogueAct,
    get_active_service,
)
from pytod.pytod_types.transcript import (
    BackendHint,
    BackendNotification,
    NaturalLanguageTypes,
    ProgramStatement,
    PyTODInstruction,
    Response,
    UserQuery,
)
from pytod.sgd_agent_behaviour import (
    agent_informs_task_status,
    agent_notifies_failure,
    agent_suggests_next_task_turn,
    count_agent_questions,
)
from pytod.sgd_analysis_utils import count_questions_asked
from pytod.sgd_invocations import InvocationsCache
from pytod.utils import snake_case, split_camel_case

logger = logging.getLogger(__name__)

QUERY_DB_ACCESS_TOOLS = {"len", "show", "slice"}
SUCCESSFUL_TRANSACTION_TOOL = "perform"
TASK_SUGGESTION_TOOL = "suggest"
_NO_ACTIVE_INTENT = "NONE"

TurnDict = dict[str, Any]  # dict derived from a SystemTurn

CALL_TAGS = [
    "call",
    "call_with_params",
    "call",
    "call_with_params",
    "call_with_selected_params",
    "call_with_params_and_selected_params",
    "call_with_carry_over",
]
ASSIGNMENT_TAGS = ["assignment"]


def transcript_factory() -> dict[str, Any]:
    return {
        "dialogue_id": "",
        "id": "",
        "services": [],  # services in the transcript
        "service_info": [],  # turn-level service info
        "turns": [],
        "apis": [],
        "metadata": {},
    }


def update_transcript(partial_transcript: dict[str, Any], **kwargs):
    """Update the transcript of the current conversation with additional
    information."""

    keys = ("dialogue_id", "id", "services", "turns", "apis", "metadata")
    for key in keys:
        value = kwargs.get(key, None)
        if value is None:
            continue
        match key:
            case "dialogue_id" | "id":
                partial_transcript[key] = value
            case "services":
                if value is not None and value not in partial_transcript["services"]:
                    partial_transcript[key].append(value)
            case "turns":
                assert isinstance(value, list)
                partial_transcript["turns"].extend(value)
                api: APIInfo = kwargs.get("apis", None)
                assert api is not None
                for v in value:
                    info = {
                        "next_service": kwargs.get("services"),
                        "active_services": api.active_services,
                        "active_intent": api.function,
                        "previous_intent": api.last_function_called,
                        "none_intent": api.none_intent,
                    }
                    # only one service is active on the system side
                    if v["author"] in {"System", "Response"}:
                        info["active_services"] = [info["next_service"]]
                    partial_transcript["service_info"].append(info)
                api_dict = api._asdict()
                for v in value:
                    this_turn_api = deepcopy(api_dict)
                    if v["author"] in {"System", "Response"}:
                        this_turn_api["active_services"] = [this_turn_api["service"]]
                    partial_transcript["apis"].append(this_turn_api)
                assert all(el for el in partial_transcript["service_info"])
            case "metadata":
                assert isinstance(value, dict)
                partial_transcript["metadata"].update(value)
            case "apis":
                pass
            case _:
                raise ValueError(f"Unknown key: {key}")
    assert len(partial_transcript["turns"]) == len(partial_transcript["service_info"])


def get_novelty_label(
    service: str, sgd_metadata: IndexMetadata
) -> Literal["seen", "unseen", "uncanny"] | None:
    """Mark a service as one of three categories:

    - seen: the service schema has been seen in training
    - unseen: the service has not been seen in training
     (mostly new domain - includes Services_4 which is sufficiently distinct from Services_{1,2,3})
    - uncanny: the service schema is a different implementation of a training set domain
    """
    if sgd_metadata.sgd_api_novelty is None:
        return
    novelty_map = sgd_metadata.sgd_api_novelty
    if service in novelty_map.zero_shot_services:
        return "unseen"
    elif service in novelty_map.seen_services:
        return "seen"
    else:
        return "uncanny"


class SuspendedTask(NamedTuple):
    service: ServiceName
    intent: IntentName


class APIInfo(NamedTuple):
    # the service for the next task
    service: ServiceName
    # two services may be active when
    # domain changes (user thanks/selects
    # entity and moves on to next task)
    active_services: list[ServiceName]
    function: IntentName
    function_snake: str
    novelty: Literal["seen", "unseen", "uncanny"]
    # the last user intent tool called
    last_function_called: IntentName
    # FindRestaurant -> Restaurant
    entity_type: str
    api_type: Literal["search", "transactional"]
    # current intent is a query
    is_query: bool
    # list of required slots not yet requested by
    # the agent or provided by the user
    unfilled_slots: list[SlotName]
    suspended_tasks: list[SuspendedTask]
    # list of slots with special "dontcare" value
    wildcard_slots: list[SlotName]
    # an intent that the agent can suggest upon task completion
    followup_command: IntentName
    # list of slots to be confirmed before executing the intent
    system_confirmed_slots: list[SlotName]
    # whether the original intent annotation was "NONE"
    none_intent: bool
    # whether the agent notified failure in the previous turn
    notified_failure: bool


def maybe_label_turn_as_novel(
    turn: dict[str, str | None],
    api_info: APIInfo | None,
    instruction: ProgramStatement,
):
    """Add a novelty label to assignments and call turns."""

    novelty_label: Literal["seen", "unseen", "uncanny"] = api_info.novelty
    if novelty_label is None:
        return
    turn.update({"novelty": novelty_label})


def create_turns(
    indexed_instructions: list[
        tuple[int | None, PyTODInstruction | NaturalLanguageTypes]
    ],
    interpreter_config: DictConfig,
    api_info: APIInfo | None = None,
    invocations_cache: InvocationsCache | None = None,
) -> list[dict[str, Any]]:
    turns = []
    user_query_turn = None
    for maybe_index, instruction in indexed_instructions:
        match instruction:
            case UserQuery(query):
                user_query_turn = {
                    "author": interpreter_config.user.author,
                    "query": query,
                }
                turns.append(user_query_turn)
            case ProgramStatement(expression=expression):
                assert maybe_index is not None
                author = interpreter_config.backend.actions.author
                program_statement_turn = {
                    "author": author,
                    "index": maybe_index,
                    "expression": expression,
                }
                maybe_label_turn_as_novel(program_statement_turn, api_info, instruction)
                assert (
                    user_query_turn is not None
                ), "Transcript updates should always contain a user query"
                if invocations_cache is not None:
                    invocations_cache.maybe_cache_instruction(
                        instruction, item=(user_query_turn, program_statement_turn)
                    )
                turns.append(program_statement_turn)
            case BackendHint(dialog=dialog):
                assert maybe_index is not None
                author = interpreter_config.backend.hints.author
                # hints that do not ground the agent utterance have null origin
                if not interpreter_config.backend.hints.multiple_slot_filling:
                    assert instruction.origin is not None
                turns.append(
                    {
                        "author": author,
                        "dialog": dialog,
                        "index": maybe_index,
                        "origin": instruction.origin,
                    }
                )
            case BackendNotification(dialog=dialog):
                assert maybe_index is not None
                assert (origin := instruction.origin) is not None
                author = interpreter_config.backend.notifications.author
                turns.append(
                    {
                        "author": author,
                        "dialog": dialog,
                        "index": maybe_index,
                        "origin": origin,
                    }
                )
            case Response(response=response):
                author = interpreter_config.response.author
                turns.append({"author": author, "text": response})
            case _:
                raise TypeError(f"Unknown instruction type: {type(instruction)}")
    return turns


def get_current_api(
    user_turn: Turn,
    user_turn_index: int,
    conversation: Conversation,
    prev_api: APIInfo | None,
    index_metadata: IndexMetadata,
    prev_system_turn: Turn | None = None,
    next_system_turn: Turn | None = None,
    command_collection: CommandCollection | None = None,
    label_novelty: bool = True,
) -> APIInfo:
    """Track the user's active intent."""

    def get_unfilled_slots(
        intent: IntentName,
        service: ServiceName,
        user_turn: Turn,
        command_collection: CommandCollection,
    ) -> list[SlotName]:
        """Find which required slots are not yet in the dialogue state."""
        required_arguments = command_collection.get(service, intent).required_slots
        required_slot_names = [arg.name for arg in required_arguments]
        state = user_turn.dialogue_state[service].slot_values
        return [s for s in required_slot_names if s not in state]

    def get_suspended_tasks(user_turn: Turn) -> list[SuspendedTask]:
        """Get the name of any tasks proposed by the user that the agent suspends."""

        for service, service_actions in user_turn.user_actions_dict.items():
            if UserDialogueAct.NEGATE_INTENT in service_actions:
                # only one task is suggested ever
                intent = service_actions[UserDialogueAct.NEGATE_INTENT][
                    0
                ].canonical_values[0]
                return [SuspendedTask(service=service, intent=intent)]
        return []

    def get_wildcard_slots(service: ServiceName, user_turn: Turn) -> list[SlotName]:
        """Find slot that take wildcard value "dontcare". These may be optional in
        some APIs (eg FindEvent) but are required in related APIs (eg BuyEventTickets)
        so the system must request them."""

        state = user_turn.dialogue_state[service].slot_values
        wildcard_slots = [
            slot for slot, values in state.items() if WILDCARD_VALUE in values
        ]
        return wildcard_slots

    # infer active service based on the current user turn
    active_service = get_active_service(
        Conversation(
            turns=conversation[: user_turn_index + 1], id=conversation.id, services=[]
        )
    )
    if next_system_turn is not None:
        assert active_service in next_system_turn.system_actions

    api_info: dict[
        Literal[
            "service",
            "active_services",
            "function",
            "function_snake",
            "last_function_called",
            "novelty",
            "type",
            "entity_type",
            "api_type",
            "is_query",
            "unfilled_slots",
            "suspended_tasks",
            "wildcard_slots",
            "followup_command",
            "system_confirmed_slots",
            "none_intent",
            "notified_failure",
        ]
    ] = {
        "service": active_service,
        "active_services": tuple(user_turn.dialogue_state.keys()),
    }
    if label_novelty:
        api_info["novelty"] = get_novelty_label(active_service, index_metadata)
    else:
        api_info["novelty"] = None
    active_intent = user_turn.dialogue_state[active_service].active_intent
    api_info["function"] = active_intent
    api_info["function_snake"] = snake_case(active_intent)
    api_info["last_function_called"] = (
        prev_api.function if prev_api is not None else active_intent
    )
    api_info["none_intent"] = False
    if prev_system_turn is None:
        api_info["notified_failure"] = False
    else:
        api_info["notified_failure"] = agent_notifies_failure(prev_system_turn).detected
    if active_intent == _NO_ACTIVE_INTENT:
        # look for the active service in the previous user turn
        assert user_turn_index - 1 > 0
        prev_active_service = get_active_service(
            Conversation(
                turns=conversation[: user_turn_index - 1],
                id=conversation.id,
                services=[],
            )
        )
        prev_usr_turn = conversation[user_turn_index - 2]
        active_intent = prev_usr_turn.dialogue_state[prev_active_service].active_intent
        try:
            assert active_intent != _NO_ACTIVE_INTENT
            assert active_intent == prev_api.function
        except AssertionError:
            # should only happen at the end of the conversation
            assert user_turn_index + 1 == len(conversation) - 1
            active_intent = prev_api.function
        api_info["function"] = active_intent
        api_info["function_snake"] = snake_case(active_intent)
        api_info["none_intent"] = True

    verb, maybe_objects = split_camel_case(api_info["function"])
    # hacky plural handling
    api_info["entity_type"] = maybe_objects
    if maybe_objects.endswith("s") and maybe_objects not in {"Bus"}:
        api_info["entity_type"] = maybe_objects[:-1]
    api_info["api_type"] = (
        "search"
        if api_info["function"] in index_metadata.search_intents
        else "transactional"
    )
    api_info["is_query"] = api_info["api_type"] == "search"
    api_info["unfilled_slots"] = get_unfilled_slots(
        active_intent, active_service, user_turn, command_collection
    )
    if prev_api is not None:
        api_info["suspended_tasks"] = prev_api.suspended_tasks + get_suspended_tasks(
            user_turn
        )
    else:
        api_info["suspended_tasks"] = []
    api_info["wildcard_slots"] = get_wildcard_slots(active_service, user_turn)
    command = command_collection.get(api_info["service"], api_info["function"])
    api_info["followup_command"] = command.followup_command
    api_info["system_confirmed_slots"] = []
    if command.system_confirmed_slots is not None:
        api_info["system_confirmed_slots"] = command.system_confirmed_slots
    return APIInfo(**api_info)


def get_metadata(leaf: list[str], prefix: ConversationPrefix) -> dict[str, Any]:
    truncated = prefix.last_turn_idx != -1
    metadata = {
        "conversation_structure": {
            "original_label": leaf,
            "new_label": leaf
            if "single_intent" in leaf or not truncated
            else "unknown",  # label changes after prefix extraction
        },
        "truncated": truncated,
        "single_turn": len(prefix.prefix) == 1,
        "additional_labels": [],
    }
    return metadata


def update_metadata(metadata: dict[str, Any], *, leaf: list[str]):
    """In-place metadata update. Currently used to assign the most
    detailed conversation structure label to a conversation."""
    metadata["conversation_structure"] = leaf


def maybe_update_conversation_structure_label(
    conversation_id: str,
    leaf: list[str],
    seen_conversations: dict[ShardName, set[DialogueID]],
    sgd_shards: dict[ShardName, list[dict[str, Any]]],
    shard_name: ShardName,
):
    """Adds a more detailed conversation structure label to a conversation. This
    is possible only if the conversation has not been truncated and the PyTOD algorithm
    is configured to traverse the entire SGD conversation structure tree as opposed to
    specified nodes.

    Notes
    -----
    The SGD conversation structure tree is found in resources/sgd/index."""

    if conversation_id in seen_conversations[shard_name]:
        convo_ref = [
            dial for dial in sgd_shards[shard_name] if dial["id"] == conversation_id
        ]
        # should be unique, otherwise it means the same conversation is split into
        # multiple nodes in the tree & we process it multiple times (ie data
        # duplication)
        assert len(convo_ref) == 1
        structure_label = convo_ref[0]["metadata"]["conversation_structure"]
        prev_leaf_len = (
            len(structure_label) if isinstance(structure_label, list) else float("inf")
        )
        if len(leaf) > prev_leaf_len:
            logger.info(
                f"Updated metadata for conversation {conversation_id} to {leaf}. "
                f"Was {structure_label}"
            )
            update_metadata(convo_ref[0]["metadata"], leaf=list(leaf))


def count_turns(data: dict[ShardName, list[dict]]):
    """Counts the total number of turns in `data`."""

    counts = 0
    for shard, shard_dialogues in data.items():
        for dial in shard_dialogues:
            counts += len(dial["turns"])

    logger.info(f"{counts} turns gathered for this split")


def extract_dialogue_history(
    conversation: Conversation, transcript: dict[str, Any], schema: CommandCollection
) -> list[dict[str, Any]]:
    """Create a list of user turns and system responses with additional metadata for
    evaluation purposes."""

    def update_metadata(
        conversation: Conversation,
        call_status_info: defaultdict[Literal["FAILURE", "SUCCESS"], set[int]],
        agent_suggested_tasks: defaultdict[Literal["YES"], set[int]],
    ):
        """Collect ground truth information about task/success failure
        to enable inserting required signal or system turns during evaluation."""
        for i, turn in enumerate(conversation):
            match turn.author:
                case Author.SYSTEM:
                    if agent_informs_task_status(turn):
                        if agent_notifies_failure(turn):
                            call_status_info["FAILURE"].add(i)
                        else:
                            call_status_info["SUCCESS"].add(i)
                    if agent_suggests_next_task_turn(turn):
                        agent_suggested_tasks["YES"].add(i)

    call_status_info = defaultdict(set)
    agent_suggested_tasks = defaultdict(set)
    update_metadata(conversation, call_status_info, agent_suggested_tasks)
    dialogue_history = []
    user_turn_idx = -2
    for transcript_idx, turn in enumerate(transcript["turns"]):
        match (author := turn["author"]):
            case "User" | "Response":
                if author == "User":
                    user_turn_idx += 2
                turn = deepcopy(turn)
                service = transcript["service_info"][transcript_idx]["next_service"]
                active_services = transcript["service_info"][transcript_idx][
                    "active_services"
                ]
                active_intent = transcript["service_info"][transcript_idx][
                    "active_intent"
                ]
                none_intent = transcript["service_info"][transcript_idx]["none_intent"]
                commands = schema.get_service_commands(service)
                apis = [cmd.name for cmd in commands]
                turn["apis"] = apis
                turn["next_service"] = service
                turn["active_services"] = active_services
                turn["active_intent"] = active_intent
                turn["none_intent"] = none_intent
                turn["call_status"] = None
                turn["make_suggestion"] = None
                turn["expected_slots"] = []
                if author == "Response":
                    for key in ("FAILURE", "SUCCESS"):
                        if len(dialogue_history) in call_status_info[key]:
                            turn["call_status"] = key
                    if len(dialogue_history) in agent_suggested_tasks["YES"]:
                        turn["make_suggestion"] = "YES"
                if transcript_idx > 0 and author == "User":
                    prev_sys_turn = conversation.turns[user_turn_idx - 1]
                    if count_agent_questions(prev_sys_turn) > 0:
                        turn["expected_slots"] = [
                            a.slot
                            for a in prev_sys_turn.system_actions[service]
                            if a.act == SystemDialogueAct.REQUEST
                        ]
                dialogue_history.append(turn)
    return dialogue_history


def get_dst_transcript(
    conversation: PyTODConversation, lowercase: bool = True
) -> PyTODConversation:
    """Remove the hint and say turns from the conversation. This utility is used
    for creating testing transcripts."""

    def qa_turn(turn: SystemTurn) -> bool:
        """Hack to determine whether the system only answers
        user questions in the current turn."""
        assert turn.get_tool_name() == "say"
        if not turn.expression[0].positional_args:
            return False
        return all("." in arg for arg in turn.expression[0].positional_args)

    def conversation_ends(turn: SystemTurn) -> bool:
        """Determine whether `say` marks conversation end."""
        assert turn.get_tool_name() == "say"
        return len(turn.expression[0].positional_args) == 0

    def offers_entity(turn: SystemTurn, current_idx: int) -> bool:
        """Determine whether `say` communicates entity info to the user."""
        assert turn.get_tool_name() == "say"
        n_positionals = len(turn.expression[0].positional_args)
        if n_positionals == 0 or n_positionals > 1:
            return False
        var = turn.expression[0].positional_args[0]
        idx = int(var[1:])
        for t in reversed(conversation.turns[:current_idx]):
            try:
                if t.index == idx:
                    ref_turn = t
                    break
            except AttributeError:
                continue
        else:
            # this should never happen
            assert False
        try:
            ref_tool = ref_turn.expression.get_tool_name()
        except AttributeError:
            # can be a hint followed by say(var)
            return False
        result = ref_tool in {"len", "show", "next", "slice"}
        if result:
            try:
                assert conversation.turns[current_idx - 1].get_tool_name() in {"next"}
            except AssertionError:
                if conversation.service_info is not None:
                    assert (
                        conversation.service_info[current_idx - 1].next_service
                        in SKIP_QUERY_BINDING
                    )
        return result

    dst_turns, turn_intents, turn_service_info = [], [], []
    for i, turn in enumerate(conversation.turns):
        skipped = False
        match turn.author:
            case "User" | "Response":
                dst_turns.append(deepcopy(turn))
            case "System":
                tool_name = turn.get_tool_name()
                match tool_name:
                    # for DST transcripts we include only
                    # `say` instructions that are not related
                    # to action selection
                    case "say":
                        if (
                            qa_turn(turn)
                            or conversation_ends(turn)
                            or offers_entity(turn, i)
                        ):
                            dst_turns.append(deepcopy(turn))
                        else:
                            skipped = True
                    case "suggest":
                        adjacent_turns = [
                            conversation.turns[i - 1],
                            conversation.turns[i + 1],
                        ]
                        [adjacent_hint] = [
                            t for t in adjacent_turns if t.author == "Hint"
                        ]
                        # if the hint to prompt the user for the next task
                        # does not have origin, it means that the agent
                        # suggests a task and so it is not transient in
                        # the dialogue history
                        if adjacent_hint.origin is not None:
                            skipped = True
                        else:
                            turn = deepcopy(turn)
                            dst_turns.append(turn)
                    case _:
                        turn = deepcopy(turn)
                        dst_turns.append(turn)
            case "Hint":
                skipped = True
            case "Signal":
                turn = deepcopy(turn)
                dst_turns.append(turn)
            case _:
                raise ValueError(f"Unknown author: {turn.author}")

        if not skipped:
            if conversation.intents is not None:
                turn_intents.append(conversation.intents[i])
            if conversation.service_info is not None:
                turn_service_info.append(conversation.service_info[i])

    # we have to re-assign the indices to ensure they are correct
    reassigned_dst_turns = []
    var_mapper = VariableMapper(0)
    for turn in dst_turns:
        match turn.author:
            case "User":
                if lowercase:
                    turn.query = turn.query.lower()
                reassigned_dst_turns.append(turn)
            case "Response":
                if lowercase:
                    turn.text = turn.text.lower()
                reassigned_dst_turns.append(turn)
            case "System":
                turn_dict = turn.model_dump()
                turn_dict.pop("novelty")
                var_mapper.add_mapping(f"x{turn.index}")
                new_expression = str(var_mapper.apply(turn.expression))
                if lowercase:
                    new_expression = new_expression.lower()
                turn_dict["expression"] = new_expression
                turn = SystemTurn.model_validate(turn_dict)
                reassigned_dst_turns.append(turn)

            case "Signal":
                if lowercase:
                    turn.dialog = turn.dialog.lower()
                reassigned_dst_turns.append(turn)
                var_mapper.add_mapping(f"x{turn.index}")
            case _:
                raise ValueError(f"Unknown author: {turn.author}")
    for var in (turn_intents, turn_service_info):
        if var:
            assert len(reassigned_dst_turns) == len(var)

    turn_idx = 0
    for turn in reassigned_dst_turns:
        match turn.author:
            case "System" | "Signal":
                turn.index = turn_idx
                turn_idx += 1
                if turn.author == "Signal":
                    origin_var = var_mapper._apply(f"x{turn.origin}")  # noqa
                    turn.origin = int(origin_var[1:])

    dst_conversation_dict = {
        "id": conversation.id,
        "turns": reassigned_dst_turns,
        "intents": None if conversation.intents is None else turn_intents,
        "service_info": None
        if conversation.service_info is None
        else turn_service_info,
        "metadata": conversation.metadata,
    }
    return PyTODConversation.model_validate(dst_conversation_dict)
