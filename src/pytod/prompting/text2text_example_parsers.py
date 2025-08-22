#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import dataclasses

from pydantic import BaseModel, ConfigDict

from pytod.command import ArgumentDefinition, ServiceCommand
from pytod.pytod_types.aliases import IntentName, ServiceName, SlotName, SlotValue
from pytod.pytod_types.pytod import AnyTurn, Intent, SystemTurn
from pytod.pytod_types.sgd_conversation import Author, Turn


class InferenceConversationExample(BaseModel):
    id: str
    source_turns: list[AnyTurn]
    oracle_target_turns: list[SystemTurn] | None = None
    target_turns: list[SystemTurn] | None = None
    oracle_prompt: str | None = None
    service: str
    apis: list[Intent] | list[ServiceCommand]
    intent: str | None = None
    none_intent: bool | None = None
    datapoint_id: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclasses.dataclass
class ConversationExample:
    id: str
    action_ids: list[str]
    source_turns: list[AnyTurn]
    target_turns: list[SystemTurn]
    service: ServiceName
    intent: IntentName
    current_task: ServiceCommand
    apis: list[Intent] | list[ServiceCommand]
    tool_name: str | None = None
    none_intent: bool | None = None


@dataclasses.dataclass
class NLUConversationExample:
    id: str
    service: ServiceName
    # conversation history
    source_turns: list[Turn]
    # slots mentioned by the user in the last turn
    target: dict[ArgumentDefinition, SlotValue]
    # slots whose value is not known
    unknown_val_slots: list[ArgumentDefinition]

    def __str__(self) -> str:
        source = []
        for turn in self.source_turns:
            prefix = "user: " if turn.author == Author.USER else "agent: "
            source.append(f"{prefix}{turn.text}\n")
        source_str = f"conversation:\n{''.join(source)}"
        targets = [f"{k.name}={v}" for k, v in self.target.items()]
        target_str = f"answer: {', '.join(targets)}"
        unknown_str = f"unknown: {', '.join([s.name for s in self.unknown_val_slots])}"
        return f"{source_str}\n{target_str}\n{unknown_str}"
