#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from copy import deepcopy
from enum import Enum
from typing import Any, Iterator, Literal, Optional, Union

from pydantic import BaseModel

from pytod.command import ServiceCall, ServiceCallResult
from pytod.pytod_types.aliases import ServiceIntent, ServiceName, SlotName
from pytod.utils import drop_empty_value_keys

_SGD_USER_NAME = "USER"
_SGD_AGENT_NAME = "SYSTEM"
_SGD_SERVICE_CALL_KEY = "service_call"
_SGD_SERVICE_RESULTS_KEY = "service_results"

NO_ACTIVE_INTENT = "NONE"

logger = logging.getLogger(__name__)


class Author(Enum):
    USER = "user"
    SYSTEM = "agent"


class DialogueAct(Enum):
    pass


class UserDialogueAct(DialogueAct):
    # common for both user and system
    INFORM = "INFORM"
    REQUEST = "REQUEST"
    GOODBYE = "GOODBYE"

    # USER SPECIFIC
    INFORM_INTENT = "INFORM_INTENT"
    AFFIRM_INTENT = "AFFIRM_INTENT"  # agree to the intent offered by agent
    NEGATE_INTENT = "NEGATE_INTENT"  # negate intent offered by agent
    AFFIRM = "AFFIRM"  # agree to the system proposal
    NEGATE = "NEGATE"  # deny agent proposal
    # Select a result being offered by the system.
    # The corresponding action may either contain no parameters,
    # in which case all the values proposed by the system are being accepted,
    # or it may contain a slot and value parameters, in which case the specified
    # slot and value are being accepted.
    SELECT = "SELECT"
    # Ask for more results besides the ones offered. Slot & vals are always empty
    REQUEST_ALTS = "REQUEST_ALTS"
    THANK_YOU = "THANK_YOU"
    # special act, not included in the SGD annotations that marks the slots in
    # the dialogue state at a given intent/service boundary turn that the
    # user does not restate but form part of the API the system should invoke next
    CARRY_OVER = "CARRY_OVER"


class SystemDialogueAct(DialogueAct):
    # common for both user and system
    INFORM = "INFORM"
    REQUEST = "REQUEST"
    GOODBYE = "GOODBYE"

    #  Confirm the value of a slot before making a transactional service call.
    CONFIRM = "CONFIRM"
    # Offer a certain value for a slot to the user.
    # The corresponding action always contains a slot and a list of values
    # for that slot offered to the user.
    OFFER = "OFFER"
    # Offer a new intent to the user. Eg, "Would you like to reserve a table?".
    # The corresponding action always has "intent" as the slot,
    # and a single value containing the intent being offered.
    # The offered intent belongs to the service corresponding to the frame.
    OFFER_INTENT = "OFFER_INTENT"
    #  Inform the user that their request was successful.
    #  Slot and values are always empty in the corresponding action.
    NOTIFY_SUCCESS = "NOTIFY_SUCCESS"
    #  Inform the user that their request was successful.
    #  Slot and values are always empty in the corresponding action.
    NOTIFY_FAILURE = "NOTIFY_FAILURE"
    # Inform the number of items found that satisfy the user's request.
    # The corresponding action always has "count" as the slot,
    # and a single element in values for the number of results obtained by the system.
    INFORM_COUNT = "INFORM_COUNT"
    REQ_MORE = "REQ_MORE"  # Asking the user if they need anything else.


class APISlotCarryOverInfo(BaseModel):
    # SGD intent name from which slot is carried over
    function: str
    # service from which slot is carried over
    service: str
    # if the value is back-tracked to multiple
    # intents in the history, this helps identify
    # the most recent intent (1 represents the
    # previous intent)
    intent_distance: int
    # the name of the slot from which the value is carried over
    slot: str
    values: list[str]


class SelectedSlotIndex(BaseModel):
    # this is the index of the slot value selected by the
    # user from multiple values proposed by the system
    index: int
    mentioned_in_utterance: bool


class SlotCarryOverValues(BaseModel):
    values: list[str]


class DialogueState(BaseModel):
    slot_values: dict[str, list]
    active_intent: str
    requested_slots: list[str]
    # for turns where the service or the intent changes,
    # this stores any relevant slots the user mentioned
    # before (in the context of a different service or
    # different intent). To populate this, the
    # conversation object has to be built with a special
    # builder, see builder function documentation below.
    slots_carried_over: Optional[
        Union[dict[str, list[APISlotCarryOverInfo]], dict[str, SlotCarryOverValues]]
    ] = None

    def get_slot_difference(
        self, new_slot_mentions: dict[SlotName, list[str]]
    ) -> Optional[dict[SlotName, list[str]]]:
        """Calculate the difference between the state at a given turn and `new_slot_mentions`.
         This helps determine which slots in the state were mentioned in a previous turn.

        Parameters
        ----------
        new_slot_mentions
            Slots mentioned by user at the current turn (annotated with UserDialogueAct.INFORM).
        """

        current_state = self.slot_values.items()
        carry_over = {}
        for slot, val_list in current_state:
            if slot not in new_slot_mentions:
                carry_over[slot] = deepcopy(val_list)
        return carry_over or None


class SystemAction(BaseModel):
    slot: str
    act: SystemDialogueAct
    values: list[str] = []
    canonical_values: list[str] = []
    metadata: Optional[dict[str, Any]] = None


class UserAction(BaseModel):
    slot: str
    act: UserDialogueAct
    values: list[str] = []
    canonical_values: list[str] = []
    # this field is populated with additional information
    # required for variable references in slot values ('carryover')
    # and list indexing expressions ('slot_selection')
    metadata: Optional[
        dict[
            Literal["carryover", "slot_selection"],
            Union[list[APISlotCarryOverInfo], SelectedSlotIndex],
        ]
    ] = None


class Turn(BaseModel):
    author: Author

    # Should be set for author = user or system.

    # For both user & system turns, this is a user utterance.
    text: Optional[str] = None

    # A list of equivalent `text` options.  Can be set for system turns.
    # Evaluations can be run against any of its elements, relaxing the need for `text`
    # to meet evaluation criteria exactly. e.g.
    #  [
    #   "I have booked you in at Nando's tonight at 7PM"
    #   "Nando's sorted for tonight at 7PM"
    #  ]
    alternative_texts: Optional[list[str]] = None
    # Should be set for user turns. For some turns where
    # the service changes, states in both services are included.
    dialogue_state: Optional[dict[ServiceName, DialogueState]] = None
    # Should be set for system turns.
    # Some boundary turns contain actions for two services
    system_actions: Optional[dict[ServiceName, list[SystemAction]]] = None
    # Should be set for system turns.
    # Some boundary turns contain actions for two services
    user_actions: Optional[dict[ServiceName, list[UserAction]]] = None
    user_actions_dict: Optional[
        dict[ServiceName, dict[UserDialogueAct, list[UserAction]]]
    ] = None
    system_actions_dict: Optional[
        dict[ServiceName, dict[SystemDialogueAct, list[SystemAction]]]
    ] = None
    service_call: Optional[
        ServiceCall
    ] = None  # Should be set for some turns when author = system
    service_results: Optional[
        ServiceCallResult
    ] = None  # should be set for system turns where a service call is made
    sgd_turn_idx: int | None = None

    def text_options(self) -> Iterator[str]:
        if self.text is not None:
            yield str(self.text)

        if self.alternative_texts is not None:
            for t in self.alternative_texts:
                yield str(t)

    def __str__(self) -> str:
        return f"{self.author}: {self.text}"


class ValuesInfo(BaseModel):
    # the slot values, as annotated in the
    # frame dialogue action "values" property
    values: list[str]
    # the idx of the turn where the values were
    # mentioned
    mentioned_in_turn: int


class SlotsCarriedOver(BaseModel):
    # ServiceIntent: the service&intent where slot values are carried over from another service
    # SlotName: the name of the slot in the ServiceIntent schema that inherits the value
    # APISlotCarryOverInfo: info about the source service, intent and slot name where value is
    #  mentioned
    carryover_slots: dict[ServiceIntent, dict[SlotName, list[APISlotCarryOverInfo]]]


class SlotMentions(BaseModel):
    """Tracks, at dialogue level, mentions of each slot by both user and system
    in each intent and service."""

    # First key: `{service}.{intent}` where `service` and `intent`
    # are service and intent names as in the SGD schema
    # Second key: a slot name, as in the (possibly transformed)
    # SGD schema
    # Values: lists of slot value info - this is because
    #  values are list annotated and a slot may be mentioned
    #  multiple times
    user: dict[ServiceIntent, dict[SlotName, list[ValuesInfo]]]
    system: dict[ServiceIntent, dict[SlotName, list[ValuesInfo]]]


class InformationProvided(BaseModel):
    # mapping containing all the slots provided by system
    # in response to user questions
    provided_slot_values: dict[ServiceIntent, dict[SlotName, list[ValuesInfo]]]


class Conversation(BaseModel):
    turns: list[Turn]
    id: str  # SGD dialogue ID
    # services in this conversation
    services: list[ServiceName]
    # track slot mentions for both user and system
    slot_mentions: Optional[SlotMentions] = None
    # track slots carried over across intents and services
    slots_carried_over: Optional[SlotsCarriedOver] = None
    # track answers to user questions
    information_provided: Optional[InformationProvided] = None

    def next_author(self) -> Author:
        """What the author of the next turn should be."""
        raise NotImplementedError("next_author method is not implemented")

    def prefix_conversations(
        self, last_speaker: Optional[Literal[Author.USER, Author.SYSTEM]] = None
    ) -> Iterator["Conversation"]:
        """Generates the sub-conversations of this conversation that are its prefixes.

        This includes the whole conversation.
        """
        for i in range(len(self.turns)):
            turns = self.turns[: (i + 1)]
            if last_speaker == Author.USER and turns[-1].author == Author.SYSTEM:
                continue
            if last_speaker == Author.SYSTEM and turns[-1].author == Author.USER:
                continue
            yield Conversation(turns=turns, id=self.id, services=self.services)

    def __len__(self) -> int:
        return len(self.turns)

    def __getitem__(self, index: int) -> Turn:
        return self.turns[index]

    def __iter__(self) -> Iterator[Turn]:
        return iter(self.turns)

    def current_turn(self) -> Optional[Turn]:
        if self.turns:
            return self.turns[-1]


def _get_turn_actions(
    turn: Turn, author_key: str = None
) -> dict[ServiceName, Union[list[UserAction], list[SystemAction]]]:
    """Extract the actions from a user/system turn."""
    assert author_key in [None, "system_actions", "user_actions"]
    if author_key is None:
        if turn.author == Author.SYSTEM:
            author_key = "system_actions"
        else:
            author_key = "user_actions"
    return getattr(turn, author_key)


def get_turn_active_services(turn: Turn) -> list[ServiceName]:
    """Get the list of services that are active at the current turn.
    More than one service is active if the user selects a result and
    then carries on to do something else in the same turn.
    """
    return list(_get_turn_actions(turn).keys())


def get_active_service(conversation: Conversation) -> ServiceName:
    """Get the service currently active in the conversation."""
    last_turn = conversation.current_turn()

    active_now = get_turn_active_services(last_turn)
    assert isinstance(active_now, list)
    if len(active_now) > 1:  # boundary frame
        assert len(conversation.turns) > 2
        prev_turn_active = get_turn_active_services(conversation.turns[-3])
        new_services = list(set(active_now).difference(prev_turn_active))
        msg = ""
        try:
            assert len(new_services) == 1
        except AssertionError:
            msg = f"{conversation.id}: Same two services active in consecutive turns."
            assert len(conversation) >= 5
            assert sorted(prev_turn_active) == sorted(active_now)

            active_two_turns_ago = list(
                set(active_now).difference(
                    get_turn_active_services(conversation.turns[-5])
                )
            )
            new_services = list(set(active_now).difference(active_two_turns_ago))
            assert len(new_services) == 1
        if msg:
            logger.debug(msg)
        return new_services[0]
    return active_now[0]


def extract_slots(
    turn: Turn,
    service: ServiceName,
    act: Union[UserDialogueAct, SystemDialogueAct],
    author: Author = None,
    enforce_values_present: bool = True,
) -> dict[str, list[str]]:
    """Extract slots/slot-value pairs which appear in the `service` frame
    of the current turn and match dialogue act `act`.

    Parameters
    ----------
    enforce_values_present
        If `True`, slots with empty value lists are not included in the
        slot mentions map.
    """

    def _sanity_check():
        assert "" not in slot_mentions
        if author is None:
            return
        if author == Author.USER:
            assert all(isinstance(a.act, UserDialogueAct) for a in actions)
        else:
            assert all(isinstance(a.act, SystemDialogueAct) for a in actions)

    actions = (
        turn.user_actions[service]
        if turn.author == Author.USER
        else turn.system_actions[service]
    )
    # a.slot condition ensures we don't end up with empty string as key when
    # processing SELECT dialogue actions that are not parametrised
    slot_mentions = {
        a.slot: deepcopy(a.values) for a in actions if a.act == act and a.slot
    }
    if enforce_values_present:
        drop_empty_value_keys(slot_mentions)
    _sanity_check()
    return slot_mentions


def get_prev_mentioned_slots(
    turn: Turn, service: ServiceName
) -> Optional[dict[str, list[str]]]:
    """Return states that appear in the current dialogue state but were
    informed beforehand."""
    assert turn.author == Author.USER
    state = turn.dialogue_state[service]
    new_slot_mentions = extract_slots(
        turn, service, UserDialogueAct.INFORM, author=Author.USER
    )
    new_slot_mentions.update(
        extract_slots(
            turn,
            service,
            UserDialogueAct.SELECT,
            author=Author.USER,
            enforce_values_present=True,
        )
    )
    return state.get_slot_difference(new_slot_mentions)

