#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import defaultdict
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Type, cast

import hydra
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from pytod.assignments import Assignments
from pytod.index_utils import get_subtrees
from pytod.interpreter.assertions import (
    assert_on_multi_frame_turns_policy,
    assert_prefix_correct,
    assert_slot_relations_correct,
    assert_turn_valid,
)
from pytod.interpreter.interpreter import TemplatesCollection, interpret_actions
from pytod.iterators import IndexMetadata, turn_pair_iterator
from pytod.preprocessing_pipelines import no_op_prefix_filter
from pytod.preprocessing_utils import ConversationPrefix
from pytod.pytod_conversation_builder import (
    parse_as_pytod_conversation,
    remove_example_invocations_key_name,
)
from pytod.pytod_text_formatter import PyTODTextFormatter
from pytod.pytod_types.pytod import Intent, PyTODConversation, ServiceInfo
from pytod.pytod_types.sgd_conversation import Turn
from pytod.pytod_types.transcript import Response, UserQuery
from pytod.sgd_invocations import (
    DiverseInvocationsSampler,
    InvocationsCache,
    collect_invocations,
)
from pytod.sgd_policy_assertions import assert_on_implicit_termination
from pytod.toolbox.toolbox import get_command_name, save_toolbox
from pytod.transcript import (
    count_turns,
    create_turns,
    extract_dialogue_history,
    get_current_api,
    get_dst_transcript,
    get_metadata,
    maybe_update_conversation_structure_label,
    transcript_factory,
    update_transcript,
)
from pytod.utils import (
    copy_schema,
    default_to_regular,
    get_sgd_shard_name,
    load_json,
    save_json,
    set_seed_no_gpu,
    sort_shards,
    typed_partial,
    write_shards,
)

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    return str(resources.files("pytod.configs.pytod") / ".")


@hydra.main(config_name="config", config_path=get_config_path())
def create_pytod_dialogues(cfg: DictConfig):
    def assert_api_info_correct(prev_user_turn: Optional[Turn]):
        try:
            assert (
                api_info.function
                == user_turn.dialogue_state[api_info.service].active_intent
            )
        except AssertionError:
            assert user_turn.dialogue_state[api_info.service].active_intent == "NONE"
            assert prev_user_turn is not None
            assert (
                api_info.function
                == prev_user_turn.dialogue_state[api_info.service].active_intent
            ) or prev_user_turn.dialogue_state[api_info.service].active_intent == "NONE"

    def assert_all_turns_converted(
        prefix: ConversationPrefix, pytod_conversation: PyTODConversation
    ):
        expected = len(prefix.prefix)
        actual = 0
        for turn in pytod_conversation.turns:
            match turn.author:
                case "User" | "Response":
                    actual += 1
        assert expected == actual

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed_no_gpu(cfg.random)
    o_path = Path(cfg.output_dir)
    schema_path = o_path / "schema.json"

    if cfg.schema_transformations[cfg.split] is None:
        copy_schema(cfg.schema_path, o_path)
    else:
        logger.info(f"Applying transformations to {cfg.split} schema")
        schema = load_json(cfg.schema_path)
        pipeline = instantiate(cfg.schema_transformations[cfg.split])
        schema = pipeline(schema)
        logger.info(f"Saving schema to {o_path}")
        save_json(schema, schema_path)
        cfg.update(schema_path=schema_path)

    # add toolbox using transformed schema
    sgd_assistant_schema = instantiate(cfg.command_collection, schema_path=schema_path)
    # TODO: AUTOMATICALLY REMAP SLOT METADATA TO KEEP UP WITH THE SCHEMA
    assert_slot_relations_correct(
        instantiate(cfg.sgd_metadata.slot_relations), sgd_assistant_schema
    )
    refined_intent_format_toolbox = []
    # add original toolbox
    # toolbox = instantiate(cfg.toolbox.toolbox)(schema_path=schema_path)
    # we call this toolbox refined because it contains modified return types compared
    # to the original SGD schema
    refined_toolbox = instantiate(cfg.toolbox.toolbox_refined)(schema_path=schema_path)
    index_path = Path(cfg.tree.index_path)
    index_info: dict[str, Any] = load_json(index_path / "metadata.json")
    index_info.update({"sgd_api_novelty": instantiate(cfg.sgd_api_novelty)})
    index_metadata = IndexMetadata.model_validate(index_info)
    subtrees = get_subtrees(**dict(cfg.tree), schema_path=schema_path)
    filtering_pipeline = instantiate(cfg.filters.scope_filters)[cfg.split]
    if (prefix_extractor := cfg.filters.prefix_extractor) is not None:
        get_prefix = instantiate(prefix_extractor)
    else:
        get_prefix = no_op_prefix_filter
    # used to display SGD conversation
    sgd_formatter = instantiate(cfg.conversation_format.formatter)  # noqa
    # used to display generated PyTOD transcripts
    pytod_formatter: PyTODTextFormatter = instantiate(cfg.pytod_formatter)
    # used to process generated PyTOD transcripts before including to dialogue history files
    sgd_shards, dialogue_histories = defaultdict(list), defaultdict(
        lambda: defaultdict(dict)
    )
    dialogue_states = defaultdict(list)
    seen_conversations = defaultdict(set)
    tool_name_formatter: Type[get_command_name] = typed_partial(
        get_command_name,
        skip_service_variations=cfg.toolbox.skip_service_variations,
        use_snake_case=cfg.toolbox.use_snake_case,
        anonymize_service=cfg.toolbox.anonymize_service,
    )
    invocations_cache = InvocationsCache(
        cfg.split, tool_name_formatter=tool_name_formatter
    )
    if (metadata_type := cfg.metadata_type) is not None:
        metadata_type = cast(
            Type[ServiceInfo] | Type[Intent], get_class(cfg.metadata_type)
        )
    for leaf, sgd_conversation in subtrees:
        assert_on_multi_frame_turns_policy(sgd_conversation)
        conversation_id = sgd_conversation.id
        shard_name = get_sgd_shard_name(conversation_id)
        should_filter = (
            filtering_pipeline(sgd_conversation)
            if filtering_pipeline is not None
            else False
        ) or conversation_id in seen_conversations[shard_name]
        if should_filter:
            maybe_update_conversation_structure_label(
                conversation_id, leaf, seen_conversations, sgd_shards, shard_name
            )
            continue
        # stores PyTOD conversation
        transcript = transcript_factory()
        # truncate conversations such that converted turns only
        # contain in-scope user/system behaviours
        prefix: ConversationPrefix = get_prefix(sgd_conversation)
        assert_prefix_correct(sgd_conversation, prefix)
        metadata = get_metadata(leaf, prefix)
        update_transcript(
            transcript,
            dialogue_id=conversation_id,
            id=f"{cfg.split}_{conversation_id}",
            metadata=metadata,
        )
        sgd_conversation_prefix = prefix.prefix
        api_info = None
        assignments = Assignments(
            interpreter_cfg=cfg.interpreter,
            dialog_id=sgd_conversation.id,
            metadata=index_metadata,
            tool_name_formatter=tool_name_formatter,
            command_collection=sgd_assistant_schema
            if cfg.interpreter.resolve_carry_over_to_entity
            else None,
        )
        if cfg.show_conversation:
            print("SGD conversation", f"{sgd_conversation.id}")
            conversation_text = sgd_formatter.conversation_to_text(sgd_conversation)
            print(conversation_text)
            print()
            print("prefix_length", len(sgd_conversation_prefix))
            print("prefix")
            print(sgd_formatter.conversation_to_text(sgd_conversation_prefix))
        prev_user_turn, prev_sys_turn = None, None
        # TODO: ASSERT AFFIRM/NEGATE COME AFTER A CONFIRMATION REQUEST
        # TODO: COULD KEEP PREV USER TURN AND PREV SYS TURN FOR VALIDATION (OR INDEX INTO THEM)
        for turn_pair_index, (user_turn, next_system_turn) in enumerate(
            turn_pair_iterator(sgd_conversation_prefix)
        ):
            # agent handles one service at a time
            assert len(next_system_turn.system_actions) == 1
            next_system_turn_copy = deepcopy(next_system_turn)
            assert_turn_valid(next_system_turn, cfg.interpreter)
            user_turn_idx = 2 * turn_pair_index
            # turns when the service changes are sometimes annotated with
            # semantic frames from both services (eg because the user selects
            # an entity and then changes the task), but only one task is active
            # at any point in time
            api_info = get_current_api(
                user_turn,
                user_turn_idx,
                sgd_conversation_prefix,
                api_info,
                index_metadata,
                prev_system_turn=prev_sys_turn,
                next_system_turn=next_system_turn,
                command_collection=sgd_assistant_schema,
            )
            assert_api_info_correct(prev_user_turn)
            invocations_cache.user_turn = user_turn
            invocations_cache.service = api_info.service
            invocations_cache.entity = api_info.entity_type
            # interpret user/system actions in the current turn pair as python
            # expressions and backend message templates
            templates_and_statements: TemplatesCollection = interpret_actions(
                cfg.interpreter, user_turn, next_system_turn, prev_sys_turn, api_info
            )
            system_response = (
                Response(response=next_system_turn.text)
                if next_system_turn is not None
                else None
            )
            user_query = UserQuery(query=user_turn.text)
            # compile the interpreted results into transcript components
            indexed_transcript_components = assignments.render_and_assign(
                user_query, templates_and_statements, api_info, system_response
            )
            # create outputs and update the transcript of the current conversation
            pytod_turns = create_turns(
                indexed_transcript_components,
                cfg.interpreter,
                api_info,
                invocations_cache=invocations_cache,
            )
            update_transcript(
                transcript,
                turns=pytod_turns,
                apis=api_info,
                services=api_info.service,
                assignments=assignments,
            )
            assert_on_implicit_termination(user_turn, prev_sys_turn, next_system_turn)
            prev_sys_turn = next_system_turn_copy
            prev_user_turn = user_turn
            logger.debug(f"Converted turn index ({user_turn_idx})")
        # parse the transcript to a standard format
        pytod_conversation = parse_as_pytod_conversation(
            transcript,
            refined_toolbox,
            refined_intent_format_toolbox,
            cfg.toolbox.toolbox_refined,
            sgd_assistant_schema,
            metadata_type,
        )
        assert_all_turns_converted(prefix, pytod_conversation)
        converted_conversation_refined_dict = pytod_conversation.model_dump()
        remove_example_invocations_key_name(
            cfg.toolbox.invocations_field_name,
            converted_conversation_refined_dict,
        )
        state_transcript = get_dst_transcript(
            pytod_conversation, lowercase=cfg.lowercase_dst_transcripts
        ).model_dump()
        dialogue_history = extract_dialogue_history(
            sgd_conversation_prefix, transcript, sgd_assistant_schema
        )
        dialogue_states[shard_name].append(state_transcript)
        dialogue_histories[shard_name][sgd_conversation_prefix.id] = dialogue_history
        sgd_shards[shard_name].append(converted_conversation_refined_dict)
        seen_conversations[shard_name].add(conversation_id)
        if cfg.show_conversation:
            if len(converted_conversation_refined_dict["turns"]) > 2:
                print("PyTOD format")
                print(pytod_formatter.conversation_to_text(pytod_conversation))
    count_turns(sgd_shards)
    sort_shards(sgd_shards)
    write_shards(sgd_shards, o_path)
    sort_shards(dialogue_states)
    write_shards(dialogue_states, Path(cfg.testing_resource_pth))
    write_shards(
        default_to_regular(dialogue_histories), Path(cfg.evaluation_resource_pth)
    )
    # optionally collect intent turns from the corpus for prompting purposes
    sampler = DiverseInvocationsSampler(invocations_cache)
    collect_invocations(cfg, invocations_cache, refined_intent_format_toolbox, sampler)
    invocations_cache.save(o_path)
    invocations_cache.save(Path(cfg.testing_resource_pth))
    save_toolbox(cfg.toolbox, o_path, refined_toolbox, refined_intent_format_toolbox)


# TODO: FOR EVERY SERVICE, DETERMINE WHETHER THE QUESTIONS FOLLOWING A TRANSACTION
#  SHOULD REFER TO THE ENTITY THAT WAS INPUT OR TO THE RESULT OF THE PERFORM ACTION
#   ANALYSE: 51_00081, 61_00055, 90_00110 (train)
# TODO: CHECK QUERIES UPGRADE WITH 9_00082 (test)
# TODO: REQUEST DISAMBIGUATION: discuss 15_00009 (dev), 30_00010 (test) examples
# TODO: DISCUSS: SHOULD CARRY-OVER BE FROM Buses or Events in "67_00092" (train)

if __name__ == "__main__":
    create_pytod_dialogues()
