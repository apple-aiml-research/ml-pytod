#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import Counter
from copy import deepcopy
from operator import itemgetter
from typing import Generic, NamedTuple, Optional, Type

from pytod.command import ServiceCommand
from pytod.execution.policy_utils import HintGenerator
from pytod.pytod_types.aliases import CanonicalValue, DialogueID, StateDict
from pytod.simulation._command_utils import APICallStatus, SearchError
from pytod.simulation.command import (
    WILDCARD_VALUE,
    Command,
    CommandT,
    SlotName,
    SlotValue,
    State,
    _init_args,
)
from pytod.simulation.command_argument import CommandArgument
from pytod.simulation.database import MongoDBCollection, empty_query, remove_attribute
from pytod.simulation.entities import Entity, get_entity

logger = logging.getLogger(__name__)


class QueryResults(NamedTuple):
    num_items: int
    status: APICallStatus


class QueryFeedback(NamedTuple):
    entity: Entity
    wrong_args: list[SlotName]
    missing_args: list[SlotName] | None


class SearchCommandState(State):
    """A custom descriptor for search commands which
    updates the state returned by the `State` descriptor
    with system-side tracked slots copied from the selected
    entity."""

    def __get__(self, instance: CommandT, owner: Type[CommandT]) -> StateDict:
        state: StateDict = super().__get__(instance, owner)
        # update state with system-side slots if an entity has
        #  been selected
        if instance.selected_entity is not None:
            for slot in instance.system_tracked_slots:
                value = getattr(instance.selected_entity, slot, None)
                if value is not None:
                    state["slot_values"][slot] = [value]
                else:
                    dial_id = instance._dialogue_id
                    cmd_name = instance.get_full_command_name(snake_cased=False)
                    logger.warning(
                        f"{dial_id} || {cmd_name} || "
                        f"Attempted to access property undefined for "
                        f"{instance.entity_name} entity: {slot}."
                    )
        # system tracked slots might need recasing
        self._maybe_restore_case(state["slot_values"], instance)
        # ensure wildcard values appear in state
        for slot in state["slot_values"]:
            if (
                slot in instance.wildcards
                # it can happen that a slot which is not an argument
                # of the command is set to a wildcard value - we
                # need to catch this to avoid an AttributeError on state
                and getattr(instance, slot, None) == WILDCARD_VALUE
                and slot not in instance.system_tracked_slots
            ):
                state["slot_values"][slot] = [WILDCARD_VALUE]
        return state


class SearchCommandArgument(CommandArgument, Generic[SlotValue]):
    """Descriptor for SearchCommand arguments, which resets the
    database pointer when the user corrects slot values."""

    def __set__(self, instance: CommandT, value: SlotValue):
        if self._name in instance.__dict__:
            dial_id = instance._dialogue_id
            cmd_name = instance.get_full_command_name(snake_cased=False)
            logger.debug(
                f"{dial_id} || {cmd_name} || "
                f"Resetting DB pointer. User updated parameter during search: {self._name} "
            )
            instance._reset_pointer()
        elif self._name not in instance.__dict__ and instance._call_id > 0:
            try:
                assert self._name in instance.keyword_args
            except AssertionError:
                # if a slot was missed but select was generated, we force
                # a perform call during select execution which increments
                # the call ID. If a required argument is subsequently set
                # on this command, the above assertion fails
                super().__set__(instance, value)
                return
            dial_id = instance._dialogue_id
            cmd_name = instance.get_full_command_name(snake_cased=False)
            logger.debug(
                f"{dial_id} || {cmd_name} || Resetting DB pointer. "
                f"User specified additional parameter during search: {self._name}"
            )
            instance._reset_pointer()
        super().__set__(instance, value)


class SearchCommand(Command):
    """Base class for all intents which have `is_transactional=False`
    in their schema. These intents represent search queries: when
    all the positional arguments are specified, the agent makes a call
    to an underlying database and informs the user about the top search
    results (referred to as _entities_)."""

    state = SearchCommandState()

    def __init__(self, dialogue_id: DialogueID):
        super().__init__(dialogue_id)
        # the raw container for the results of the current DB call
        self._raw_entities: list[dict[SlotName, CanonicalValue]] = []
        # the entity which the agent is going to present to the user next
        self.current_entity: Optional[Entity] = None
        # entities that have already been mentioned
        self._mentioned_entities: list[Entity] = []
        # entity that the user selected
        self.selected_entity: Optional[Entity] = None
        # a slot which can be used to establish entity
        # uniqueness (e.g., 'hotel_name', 'property_name')
        self._entity_cmp_key: list[SlotName] = []
        # a slot according to which results should be sorted
        self._result_sort_key: SlotName = ""
        # points to the current result
        self._results_pointer: int = -1
        # how many times the object was called
        self._call_id = 0
        # set during command building
        self._schema: ServiceCommand | None = None
        self._service_schema: list[ServiceCommand] | None = None
        # slots that can have any value
        self.wildcards: list[SlotName] = []

    def __getattr__(self, item: str) -> Optional[SlotValue]:
        """Handle access to undefined attributes."""
        dial_id = self._dialogue_id
        cmd = self.get_full_command_name(snake_cased=False)
        if item in self.entity_attributes:
            msg = f"{dial_id} || {cmd} || Accessed entity property by command referencing: {item}."
            if self.selected_entity is not None:
                logger.warning(msg)
                return getattr(self.selected_entity, item)
            elif self.current_entity is not None:
                logger.warning(msg)
                return getattr(self.current_entity, item)
            else:
                logger.warning(
                    f"{dial_id} || {cmd} || Attempted reading unknown attribute: {item}."
                )
                self._unknown_properties_accessed.append(item)
                return
        if item in self._unknown_properties_set:
            return self._uknown_properties_set[item]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")

    def __next__(self) -> Entity:
        """Returns the top result from the entities retrieved from
        the database."""
        self._results_pointer += 1
        logger.debug(f"Results pointer: {self._results_pointer}")
        try:
            current_entity = self._raw_entities[self._results_pointer]
        except IndexError:
            raise StopIteration()

        current_entity = get_entity(
            self.entity_name, current_entity, cmp_attributes=self._entity_cmp_key
        )
        if current_entity not in self._mentioned_entities:
            self._mentioned_entities.append(current_entity)
            self.current_entity = current_entity
            return current_entity
        else:
            logger.debug(f"Skipping entity: {current_entity}")
            # DB may return entities the agent already mentioned to the user
            #  which are not mentioned again -
            current_entity = self.__next__()
        return current_entity

    def get_entity_obj(
        self, database: MongoDBCollection, state_info: dict[SlotName, str]
    ) -> QueryFeedback:
        """Used at runtime to store predictions made so far for
        carry-over purposes if no prediction was made so far."""

        missing_args, wrong_args = None, None
        if all(arg in state_info for arg in self.positional_args):
            wrong_args = list(state_info.keys())
            self._query_database(
                database,
                parameters={
                    k: v for k, v in state_info.items() if k not in self.keyword_args
                },
            )
            if self._raw_entities:
                wrong_args = [
                    arg for arg in wrong_args if arg not in self.positional_args
                ] or None
            else:
                if not self.positional_args:
                    missing_args = list(self.keyword_args.keys()) or None
                    wrong_args = None
                else:
                    logger.warning(
                        "No entities retrieved, but all positional args were specified"
                    )
        else:
            missing_args = [
                arg for arg in self.positional_args if arg not in state_info
            ]
        properties = {
            arg: values[-1] for arg, values in self.state["slot_values"].items()
        }
        for arg in self.schema.result_slots:
            if arg not in properties:
                properties[arg] = None
        entity = get_entity(
            self.entity_name,
            attributes=properties,
            cmp_attributes=self._entity_cmp_key,
        )
        return QueryFeedback(
            entity=entity, wrong_args=wrong_args, missing_args=missing_args
        )

    def __len__(self) -> int:
        """Return the number of search results for the current query."""
        if self._raw_entities is None:
            return 0
        return len(self._raw_entities)

    def __select__(
        self,
        entity: Optional[Entity] = None,
        key: Optional[SlotName] = None,
        value: Optional[str] = None,
    ) -> Optional[Entity]:
        if key is not None:
            assert (
                value
            ), f"Selected entity by specifying key={key} but value=None. Expected value."
            if getattr(self.current_entity, key) == value:
                self.selected_entity = self.current_entity
                return self.current_entity
            else:
                for entity in self._mentioned_entities:
                    if getattr(entity, key) == value:
                        self.selected_entity = entity
                        return entity
                raise SearchError(f"No mentioned entity matched {key}={value}.")
        if entity is not None:
            cmd_name = self.get_full_command_name(snake_cased=False)
            if entity not in self._mentioned_entities:
                raise SearchError(
                    f"{self.uuid} || {cmd_name} || " "Selected entity was not mentioned"
                )
            if entity != self.current_entity:
                logger.warning(
                    f"{self.uuid} || {cmd_name} || "
                    f"Selected entity is not the last entity mentioned. This is not expected."
                )
            self.selected_entity = entity
            return entity

    def _reset_pointer(self):
        """Set pointer to the head of the list. Called automatically
        when the user changes the value of an attribute."""
        self._results_pointer = -1

    @property
    def entity_cmp_key(self):
        """An entity attribute acting as comparison key between two
        entities. If not set, entities compare equal if all their
        attributes are equal."""
        return self._entity_cmp_key

    @entity_cmp_key.setter
    def entity_cmp_key(self, key: SlotName):
        self._entity_cmp_key = key

    @property
    def result_sort_key(self):
        """An attribute according to which to sort search results."""
        return self._result_sort_key

    @result_sort_key.setter
    def result_sort_key(self, key: SlotName):
        self._result_sort_key = key

    def perform(self, database: MongoDBCollection) -> QueryResults:
        """Endpoint for calling the database with the current parameters.

        Returns
        -------
        An integer corresponding to the number of items retrieved from the database.
        """
        assert (
            database is not None
        ), f"Something went wrong, no context was passed to {self.name} ({self.service}) execution."
        self._query_database(database)
        self._executed = True
        return QueryResults(status=APICallStatus.SUCCESS, num_items=len(self))

    def _maybe_carry_over_wild_cards(
        self,
        query_results: list[dict],
        resolved_wildcards: list[dict[SlotName, SlotValue]],
    ):
        """Wildcard slots are carried over to other queries, unless they are system tracked slots.
        However, in our syntax up to v0.8.0 the entity properties are carried over. Therefore, we
        must ensure that the variable reference is correct."""
        for slot in [s for s in self.wildcards if s not in self.system_tracked_slots]:
            for r in query_results:
                r[slot] = WILDCARD_VALUE

    def _query_database(
        self, database: MongoDBCollection, parameters: dict[SlotName, str] | None = None
    ):
        """Call the underlying database with the specified parameters and
        store the results."""

        def filter_results(
            query_results: list[dict[SlotName, CanonicalValue]]
        ) -> list[dict[SlotName, CanonicalValue]]:
            """Ensure the query results match the dialogue annotations so that
            they can be used for DST/NLG."""

            # remove duplicate entities
            duplicates = []
            if self.entity_cmp_key:
                entities_cnt = Counter(
                    [e[self.entity_cmp_key[0]] for e in query_results]
                )
                duplicates = [key for key in entities_cnt if entities_cnt[key] > 1]
            query_results = [
                r
                for r in query_results
                if (
                    (r[self.entity_cmp_key[0]] not in duplicates)
                    or  # noqa
                    # the same entity may have different properties in subsequent calls
                    # in the same dialogue (eg number_of_rooms in the hotel domain can be
                    # different for the same hotel) so we choose the correct entity by
                    # tracking how many times was the database called
                    (
                        r[self.entity_cmp_key[0]] in duplicates
                        and r["call_id"] == self._call_id
                    )
                )
            ]
            if empty_query(parameters):
                # calls with empty parameters may return randomly sampled results.
                # if this happens multiple times in a dialogue, then we need to
                # know how many times the intent was called to return the right results
                # see(dev / 10_00054)
                query_results = [
                    r for r in query_results if r["call_id"] == self._call_id
                ]
                # used by services which accept empty queries (eg Alarm_1)
                # to ensure results returned follow annotation order and thus
                # can be used for NLG
                query_results.sort(key=itemgetter("order"))
            else:
                if self.result_sort_key:
                    query_results.sort(key=itemgetter(self.result_sort_key))
            # remove special attributes used to simulate the corpus annotations
            remove_attribute("order", query_results)
            remove_attribute("call_id", query_results)
            unique_queries = []
            for q in query_results:
                if q not in unique_queries:
                    unique_queries.append(q)
            return unique_queries

        self._call_id += 1
        if parameters is None:
            parameters = self.get_api_call_parameters()
        self.wildcards = self._get_wildcards(parameters)
        resolved_wildcard_values = self.resolve_wildcard_values(parameters)
        if (
            resolved_wildcard_values is not None
            and self._entity_cmp_key[0] in resolved_wildcard_values[0]
        ):
            resolved_wildcard_values.sort(key=lambda x: x[self._entity_cmp_key[0]])
        normalised_params = self._normalizer.normalise(self._service, parameters)
        self.log_normalisation(normalised_params)
        parameters = normalised_params.result
        parameters.update({"dialogue_id": self._dialogue_id})
        query_results = []
        # make a query for every resolved value of the parameters
        if resolved_wildcard_values is not None:
            query_params = deepcopy(parameters)
            for resolved_values_dict in resolved_wildcard_values:
                query_params.update(resolved_values_dict)
                query_results.extend(database.query(query_params))
            try:
                query_results.sort(key=lambda x: x[self.result_sort_key])
            except KeyError:
                logger.warning(f"{self.uuid}: Could not sort database results")
                query_results.sort(key=lambda x: x["order"])
            # nb: not needed for v0.8.1 and above
            self._maybe_carry_over_wild_cards(query_results, resolved_wildcard_values)
        else:
            query_results.extend(database.query(parameters))
        try:
            self._raw_entities = filter_results(query_results)
        except KeyError as e:
            logger.warning(
                f"{self.uuid}: Could not filter duplicates from database results"
                f" due to a key error: {e.args[0]}"
            )
            self._raw_entities = query_results

    def get_entity_by_idx(self, idx: int) -> Optional[Entity]:
        """Retrieve an entity by index."""
        if self._raw_entities is not None:
            try:
                raw_entity = self._raw_entities[idx]
            except IndexError:
                dial_id = self._dialogue_id
                cmd_name = self.get_full_command_name(snake_cased=False)
                logger.warning(
                    f"{dial_id} || {cmd_name} || "
                    f"Attempted to access entity at {idx}, but there are only"
                    f"{len(self._raw_entities)} entities"
                )
                return
            return get_entity(
                self.entity_name, raw_entity, cmp_attributes=self._entity_cmp_key
            )
        return

    @classmethod
    def build(
        cls: Type[CommandT],
        dialogue_id: DialogueID,
        command_schema: Optional[ServiceCommand] = None,
        service_schemata: Optional[list[ServiceCommand]] = None,
        hint_generator: Optional[HintGenerator] = HintGenerator(),
    ) -> CommandT:
        """Build a Command instance for the current dialogue session
        using the schema and the entity name.

        Parameters
        ----------
        dialogue_id
            The ID of the current dialogue session.
        command_schema
            Schema of the command, parsed from the augmented SGD schema.
        service_schemata
            Schema of other commands from the same service.
        hint_generator
            An object the command uses to generate hints that guide the
            user-agent interaction.
        """

        command = cls(dialogue_id)
        # NB: normally this would not be exposed, but we
        #  do so to facilitate experimentation with
        #  different ways to control the model
        command._hint_generator = hint_generator
        if command_schema is not None:
            if command_schema.entity_name is None:
                raise ValueError("Entity name must be specified is schema is given.")
            if (system_tracked_slots := command_schema.system_tracked_slots) is None:
                command._system_tracked_slots = ()
            else:
                command._system_tracked_slots = tuple(system_tracked_slots)
            command._entity_cmp_key = command_schema.entity_cmp_key
            command._result_sort_key = command_schema.result_sort_key
            command._entity_name = command_schema.entity_name
            command._service = command_schema.service
            _init_args(command, command_schema, service_schemata)
            command._entity_attributes = command_schema.result_slots
            command._followup_command = command_schema.followup_command
        if command._positional_args:
            assert (
                command._hint_generator is not None
            ), "Expected a hint generator for commands with positional arguments"
        return command
