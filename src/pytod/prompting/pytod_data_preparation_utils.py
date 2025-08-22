#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Type, TypeVar

from pydantic import TypeAdapter

from pytod.command import CommandCollection, ServiceCommand
from pytod.interpreter.metadata import ENTITY_SELECTION_TOOL, SPECIAL_TOOLS
from pytod.prompting.pytod_conversation_history_formatters import (
    ConversationHistoryFormatterType,
    RenderedEntitiesConversationHistoryFormatter,
)
from pytod.prompting.pytod_prompt_formatters import (
    ConversationFormatterType,
    RenderedEntitiesConversationFormatter,
    TaskStackPromptFormatter,
)
from pytod.prompting.pytod_target_formatters import TargetFormatter
from pytod.pytod_types.aliases import DialogueID, VariableName
from pytod.pytod_types.pytod import AnyTurn, PyTODConversation, ServiceInfo, SystemTurn
from pytod.transcript import (
    QUERY_DB_ACCESS_TOOLS,
    SUCCESSFUL_TRANSACTION_TOOL,
    TASK_SUGGESTION_TOOL,
    TurnDict,
)
from pytod.utils import filter_shards

T = TypeVar("T")
FormattedTarget = str
FormattedTurn = str
logger = logging.getLogger(__name__)


def get_variable_names(max_index: int) -> list[VariableName]:
    """Return a list of variable names, starting at 0 and
    ending at `max_index`."""
    return [f"x{digit}" for digit in range(max_index + 1)]


def parse_file_as(type_: Type[T], file: str | Path) -> T:
    """Simulate a pydantic v1 function for file loading and validation."""
    with Path(file).open("r") as fin:
        data = json.loads(fin.read())
    return TypeAdapter(type_).validate_python(data)


def get_conversations(
    data_dir: Path | str, return_only: set[DialogueID] | None = None
) -> Iterator[PyTODConversation]:
    """Iterator through the PyTOD conversation shards."""

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    ffs = sorted(data_dir.glob("dialogues*.json"))
    ffs, shards_to_dials = filter_shards(ffs, return_only=return_only)
    random.shuffle(ffs)
    for ff in ffs:
        shard_name = ff.name
        if "metrics" in shard_name:
            continue
        logger.info(f"Processing shard: {shard_name}")
        convos = parse_file_as(list[PyTODConversation], ff)
        for c in convos:
            if shards_to_dials is None:
                yield c
            else:
                dial_id = re.sub(r"(train_|dev_|test_)", "", c.id)
                if dial_id in shards_to_dials[shard_name]:
                    yield c


def get_turn_apis(
    schema: CommandCollection,
    service_info: ServiceInfo,
    multidomain_prompts: bool = False,
) -> list[ServiceCommand]:
    """Get the service commands which will be used for prompting the model
    at this turn."""
    if multidomain_prompts:
        apis = []
        for service in service_info.active_services:
            apis.extend(schema.get_service_commands(service))
        return apis
    return schema.get_service_commands(service_info.next_service)


def is_policy_target(target_turns: list[SystemTurn]) -> bool:
    """Determine whether `target_turns` are for DST or policy."""
    nlg_call = [t for t in target_turns if t.get_tool_name() == "say"]
    if not nlg_call:
        return False
    # only one say command expected
    assert len(nlg_call) == 1
    [cmd] = nlg_call
    # say() is considered DST target
    if not cmd.expression[0].positional_args:
        return False
    # say(x2.address) is a DST target (tracking req slots)
    if all("." in val for val in cmd.expression[0].positional_args):
        return False
    return True


def multiple_calls_in_user_turn_offset(source_turns: list[AnyTurn]) -> int:
    """This offset is relevant only if the text2text generation is
    configured to generate examples such that we can predict with rendered entities.
    In this case, when the user selects an entity and the domain changes, we have two
    training datapoints:

        - one where a `select` instruction followed by other instructions is
        predicted directly
        - one where all other instructions except `select` are predicted
        given the prompt, session transcript and the `select`

    We want a unique ID for each datapoint but we establish the datapoint ID based
    on user turn number and policy prediction offset at the moment. Hence, we
    need to count in how many user turns we had more than one example.
    """
    offset = 0
    next_system_turns = []
    for i, turn in enumerate(source_turns):
        if turn.author == "User":
            # we increment the offset by 1 if the turn is a multi-domain turn
            if next_system_turns:
                offset += int(
                    next_system_turns[0].get_tool_name() == "select"
                    and any(
                        t.get_tool_name() not in SPECIAL_TOOLS
                        for t in next_system_turns[1:]
                    )
                )
            next_system_turns = []
        if turn.author == "System":
            next_system_turns.append(turn)
    if next_system_turns:
        offset += int(
            next_system_turns[0].get_tool_name() == "select"
            and any(
                t.get_tool_name() not in SPECIAL_TOOLS for t in next_system_turns[1:]
            )
        )
    if (
        last_turn := source_turns[-1]
    ).author == "System" and last_turn.get_tool_name() == "select":
        offset += 1
    return offset


def get_datapoint_idx(
    source_turns: list[AnyTurn], target_turns: list[SystemTurn]
) -> int:
    """Get the index of the datapoint. This is the number of user turns in `source_turns`,
    incremented by 1 if `target_turns` are system actions (ie calls to NLG)."""

    if target_turns is None:
        raise ValueError("Cannot establish datapoint ID without target turns.")
    idx = -1
    for i, t in enumerate(source_turns):
        if t.author in {"User", "Response"}:
            idx += 1
    return (
        idx
        + int(is_policy_target(target_turns))
        + multiple_calls_in_user_turn_offset(source_turns)
    )


def format_source_and_targets(
    history_formatter: ConversationHistoryFormatterType,
    target_formatter: TargetFormatter,
    history: list[dict],
    transcript: list[dict[str, Any]],
    target: list[str],
    skip_targets: bool = False,
) -> tuple[list[FormattedTurn], list[FormattedTarget]]:
    """Format the source and targets included in the dialogue history according to the
    ground truth transcript for inclusion in the dialogue history file. We avoid
    doing this at runtime because errors in the predicted transcript will
    crash the target formatter. However, these targets are needed for the
    Trainer due to the way `huggingface` works.

    Parameters
    ----------
    skip_targets
        If `True`, targets are not formatted.
    """

    for turn in reversed(history):
        if "index" in turn:
            index = int(turn["index"])
            break
    else:
        index = 0

    formatted_source, var_mapper = history_formatter(
        PyTODConversation.model_validate({"id": "", "turns": history}).turns
    )
    if skip_targets:
        return formatted_source, []
    turn_models = []
    for i, t in enumerate(target, start=1):
        assert transcript[len(history) + i - 1]["expression"] == t
        turn_models.append(
            SystemTurn.model_validate(
                {"author": "System", "expression": t.strip(), "index": index + i}
            )
        )
    fmt_target = target_formatter(turn_models, var_mapper)
    fmt_target = (
        [s.strip() for s in fmt_target.split(target_formatter.expression_list_sep)]
        if target_formatter.expression_list_sep
        else [fmt_target]
    )
    return formatted_source, fmt_target


def join_target_and_source_to_history_files(
    dialogue_history: list[dict[str, Any]],
    dst_transcript: dict[str, Any],
    conversation_formatter: ConversationFormatterType,
    schema: CommandCollection,
    multidomain_prompts: bool = False,
    rendered_entities: bool = False,
):
    """Add DST targets to dialogue history files. These are used during inference for model
    output analysis. The following information is added:

        - dst_target_transcript - these are the targets where variable indices are according to
        the gold DST transcript (which includes some `say` turns in history)
        - dst_target_formatted - same targets as above, but with variable indices remapped
        to account for the absence of `say` turns in the history
        - transcript_pointers - for each target the (exclusive) index of the last turn from the gold
        DST transcript source turns
        - gold_dst_source - strings representing the gold input to the model, for which
        `dst_target_formatted` are the ground truth targets
    """

    def get_system_turns(
        start_index: int,
        state_transcript: list[dict[str, Any]],
        query: str,
    ) -> tuple[int, list[TurnDict] | None]:
        """Get the system turns before the next agent response."""
        turns = []
        next_start_index = start_index
        try:
            assert state_transcript[start_index - 1]["query"] == query
        except AssertionError:
            query = query.lower()
            assert state_transcript[start_index - 1]["query"] == query
        for i in range(start_index, len(state_transcript)):
            state_turn = state_transcript[i]
            next_start_index = i
            if state_turn["author"] == "System":
                if any(
                    state_turn["expression"].startswith(tool)
                    for tool in {SUCCESSFUL_TRANSACTION_TOOL, TASK_SUGGESTION_TOOL}
                ):
                    assert state_transcript[i + 1]["author"] == "Response"
                    continue
                else:
                    turns.append(state_turn)
            if state_turn["author"] == "Signal":
                assert state_transcript[i + 1]["author"] == "Response"
                continue
            if state_turn["author"] == "Response":
                next_start_index += 1
                break
        return next_start_index + 1, turns or None

    def get_targets_and_source_offset(
        turns: list[dict[str, Any]]
    ) -> tuple[int, list[list[str]]]:
        """Extract targets from system turns. Each element of the output
        is a target predicted before the agent communicates with the user.

        Returns
        -------
        offset
            The number of additional turns in the source transcript that need to be added
            to predict the second target
        targets
            A list containing targets that can be predicted without stopping to execute the app.
        """
        offset = 0
        if len(turns) == 1:
            return offset, [[turns[0]["expression"]]]
        targets, temp = [], []
        for turn in turns:
            if any(
                turn["expression"].startswith(tool) for tool in QUERY_DB_ACCESS_TOOLS
            ):
                targets.append(temp)
                offset = len(temp) + 1
                temp = []
            else:
                temp.append(turn["expression"])
        if temp:
            targets.append(temp)
        return offset, targets

    history_formatter = conversation_formatter.conversation_fmt
    target_formatter = conversation_formatter.target_fmt
    intent_formatter = conversation_formatter.intent_fmt
    prompt_formatter = conversation_formatter.prompt_fmt
    transcript = dst_transcript["turns"]
    lowercase = not dialogue_history[0]["query"] == transcript[0]["query"]
    assert history_formatter.start_index == 0
    start_index = 1
    for i, turn in enumerate(dialogue_history):
        if turn["author"] == "User":
            transcript_pointers = [start_index]
            query = turn["query"]
            if lowercase:
                query = query.lower()
            start_index, sys_turns = get_system_turns(start_index, transcript, query)
            pointer_offset, this_turn_targets = get_targets_and_source_offset(sys_turns)
            assert len(this_turn_targets) in range(1, 3)
            if len(this_turn_targets) == 2:
                # prediction + insert show
                transcript_pointers.append(transcript_pointers[-1] + pointer_offset)
            if multidomain_prompts:
                fmt_intents = []
                for service in turn["active_services"]:
                    fmt_intents.extend(
                        [
                            intent_formatter(
                                api,
                                n_source_turns=len(dialogue_history[:i]) + 1,
                                transcript_start_ndx=0,
                            )
                            for api in schema.get_service_commands(service)
                        ]
                    )
            else:
                fmt_intents = [
                    intent_formatter(
                        schema.get(turn["next_service"], api),
                        n_source_turns=len(dialogue_history[:i]) + 1,
                        transcript_start_nd=0,
                    )
                    for api in turn["apis"]
                ]
            # for rendered entity formatter, the active service is used
            # to determine, for example, how system offers should be
            # displayed
            if hasattr(history_formatter, "active_service"):
                history_formatter.active_service = turn["next_service"]
            formatted_targets, formatted_sources = [], []
            for i, (pointer, target) in enumerate(
                zip(transcript_pointers, this_turn_targets)
            ):
                fmt_source_pieces, fmt_target = format_source_and_targets(
                    history_formatter,
                    target_formatter,
                    transcript[:pointer],
                    transcript,
                    target,
                )
                # these are policy turns so there should be a single target domain
                if i > 0:
                    fmt_intents = [
                        intent_formatter(
                            schema.get(turn["next_service"], api),
                            n_source_turns=len(dialogue_history[:i]) + 1,
                            transcript_start_ndx=0,
                        )
                        for api in turn["apis"]
                    ]
                kwargs = {
                    "conversation_history": fmt_source_pieces,
                    "intents": fmt_intents,
                }
                if isinstance(prompt_formatter, TaskStackPromptFormatter):
                    kwargs.update(
                        {
                            "entities": history_formatter.state.entity_info,
                            "mapper": history_formatter.state.var_mapping,
                        }
                    )
                source = prompt_formatter(**kwargs)
                formatted_targets.append(fmt_target)
                formatted_sources.append(source)
                conversation_formatter.reset()
                if hasattr(history_formatter, "active_service"):
                    history_formatter.active_service = turn["next_service"]
            turn["dst_targets_formatted"] = formatted_targets
            turn["dst_targets_transcript"] = this_turn_targets
            # a list same len as targets where each integer points to the
            # last turn of the transcript included in the history to predict
            # the target at the same position
            turn["transcript_pointers"] = transcript_pointers
            turn["gold_dst_source"] = formatted_sources
        else:
            turn["dst_targets_transcript"] = None
            turn["dst_targets_formatted"] = None
            turn["transcript_pointers"] = None
            turn["gold_dst_source"] = None


class PosInfo(NamedTuple):
    # where to insert the new target/source
    insert_index: int
    # position of the select instruction within the multiline
    # instructions list - need to insert the new target
    selection_target_pos: int
    # position of the multiline instruction containing the
    # entity selection amongst the target for the turn
    target_pos: int


def join_target_and_source_to_history_files_rendered_entities(
    dialogue_history: list[dict[str, Any]],
    dst_transcript: dict[str, Any],
    conversation_formatter: RenderedEntitiesConversationFormatter,
    schema: CommandCollection,
    multidomain_prompts: bool = False,
    rendered_entities: bool = False,
):
    """A wrapper around `join_target_and_source_to_history_files` which adds
    a target to turns where the user performs selection at the same time
    with another action (eg intent change). The target consists of all the
    other instructions apart from selection, and it is added to the dataset
    so that these instructions can be predicted conditioned an a rendered
    entity."""

    def get_new_source_insert_idx(
        target: list[list[FormattedTarget]],
    ) -> None | PosInfo:
        """Return the index at which an additional source-target pair has to be
        inserted in the dialogue history."""
        # sanity check to ensure that entity selection always comes first
        for t in target[1:]:
            assert all(ENTITY_SELECTION_TOOL not in instr for instr in t)
        first_target = target[0]
        if len(first_target) < 1:
            return
        if first_target[0].startswith(ENTITY_SELECTION_TOOL):
            return PosInfo(insert_index=1, selection_target_pos=0, target_pos=0)
        return

    assert rendered_entities is True
    join_target_and_source_to_history_files(
        dialogue_history,
        dst_transcript,
        conversation_formatter,
        schema,
        multidomain_prompts,
        rendered_entities,
    )
    transcript = dst_transcript["turns"]
    history_formatter = conversation_formatter.conversation_fmt
    target_formatter = conversation_formatter.target_fmt
    intent_formatter = conversation_formatter.intent_fmt
    prompt_formatter = conversation_formatter.prompt_fmt
    # loop through the history, adding a target and the corresponding source
    # for turns where targets selection followed by one or more instructions.
    # The selection is added to the source and the remaining instructions are
    # the targets.
    for i, turn in enumerate(dialogue_history):
        if turn["author"] == "User":
            formatted_targets = turn["dst_targets_formatted"]
            transcript_targets = turn["dst_targets_transcript"]
            insert_source_target_at = get_new_source_insert_idx(formatted_targets)
            if insert_source_target_at is not None:
                # where in the list of multiline targets should the list with the new
                # targets be inserted
                insert_idx = insert_source_target_at.insert_index
                # the position of the multiline target containing the selection in the
                # list of multiline targets
                target_pos = insert_source_target_at.target_pos
                # the position of the entity selection instr in its multiline target
                selection_pos = insert_source_target_at.selection_target_pos
                formatted_targets.insert(
                    insert_idx, formatted_targets[target_pos][selection_pos + 1 :]
                )
                transcript_targets.insert(
                    insert_idx, formatted_targets[target_pos][selection_pos + 1 :]
                )
                current_pointers = turn["transcript_pointers"]
                new_pointer = current_pointers[0] + 1
                current_pointers.insert(insert_idx, new_pointer)
                history_formatter.active_service = turn["next_service"]
                fmt_source, _ = format_source_and_targets(
                    history_formatter,
                    target_formatter,
                    transcript[:new_pointer],
                    transcript,
                    [],
                    skip_targets=True,
                )
                fmt_intents = [
                    intent_formatter(
                        schema.get(turn["next_service"], api),
                        n_source_turns=len(dialogue_history[:i]) + 1,
                        transcript_start_ndx=0,
                    )
                    for api in turn["apis"]
                ]
                kwargs = {
                    "conversation_history": fmt_source,
                    "intents": fmt_intents,
                    "entities": history_formatter.state.entity_info,
                    "mapper": history_formatter.state.var_mapping,
                }
                turn["gold_dst_source"].insert(insert_idx, prompt_formatter(**kwargs))
                conversation_formatter.reset()
