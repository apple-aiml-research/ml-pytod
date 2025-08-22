#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
from collections import defaultdict
from functools import singledispatchmethod

from hydra.utils import instantiate
from omegaconf import DictConfig

from pytod.command import CommandCollection
from pytod.inference.assistant import (
    PendingRequest,
    ProcessedRequest,
    PyTODAssistant,
    PyTODAssistantFeedback,
)
from pytod.inference.constraint_prompts import (
    ConstraintTemplate,
    HallucinatedArgNameCatValueConstraintTemplate,
    HallucinatedArgNameConstraintTemplate,
    HallucinatedBoolEnumArgValueTemplate,
    MemorisedArgumentNameConstraintTemplate,
)
from pytod.inference.constraint_prompts_utils import INVALID_SLOT, MQAChoice
from pytod.inference.constraint_request import (
    ConstraintRequest,
    MemorisationInfo,
    SlotConstraintRequest,
    ValueConstraintRequest,
)
from pytod.parser.expresion_validation_utils import ConstrainedKeyword
from pytod.pytod_types.aliases import DialogueID
from pytod.utils import stringify_list

logger = logging.getLogger(__name__)


class KeywordConstraintRequest(PendingRequest):
    slot_mapping: dict[MQAChoice, ConstrainedKeyword]
    request: ConstraintRequest


class ProcessedConstraintRequest(ProcessedRequest):
    slot_mapping: dict[MQAChoice, ConstrainedKeyword]
    request: ConstraintRequest
    _constrained_keyword: ConstrainedKeyword = (None, None)
    _selected_option: str | None = None

    @property
    def selected_option(self) -> str:
        return self._selected_option

    @property
    def invalid_value_option(self) -> tuple[str, ConstrainedKeyword] | None:
        for k, v in self.slot_mapping.items():
            if v.keyword == INVALID_SLOT:
                return k, v

    @selected_option.setter
    def selected_option(self, qa_choice: str):
        try:
            self._constrained_keyword = self.slot_mapping[qa_choice]
            self._selected_option = qa_choice
        # very occasionally the LLM might generate the start of an
        # option rather than the letter
        except KeyError:
            valid_opts = stringify_list(list(self.slot_mapping.keys()), word="or")
            logger.error(
                f"An agent generated choice {qa_choice} which is "
                f"not valid. Valid options: {valid_opts}"
            )
            for k, v in self.slot_mapping.items():
                if v.keyword.startswith(qa_choice):
                    self._selected_option = k
                    self._constrained_keyword = v
                    logger.info(f"Selected keyword: {v[0]}")
                    break
            else:
                valid_keywords = stringify_list(
                    list([v.keyword for v in self.slot_mapping.values()]), word="or"
                )
                logger.error(
                    f"{qa_choice} matched no valid keyword. "
                    f"Valid keywords: {valid_keywords}"
                )
                if qa_choice == "None":
                    logger.warning(
                        "Assistant did not generate a valid option. "
                        "None was generated instead"
                    )
                    option, keyword = self.invalid_value_option
                    self._selected_option = option
                    self._constrained_keyword = keyword
                else:
                    self._selected_option = None
                    if isinstance(self.request, ValueConstraintRequest):
                        slot = self.request.argument
                    else:
                        slot = self.request.predicted_slot
                    self._constrained_keyword = ConstrainedKeyword(slot, None)

    @property
    def constrained_keyword(self) -> ConstrainedKeyword:
        return self._constrained_keyword


class SchemaSupervisor(PyTODAssistant):
    """A PyTOD agent which constrains a keyword to be one of the
    slots in the SGD schema.

    Parameters
    ----------
    optimise_prompt
        If `True` slots that have been predicted already are not
        included in the constraint prompt.
    """

    def __init__(
        self,
        pipeline_config: DictConfig,
        generation_config: DictConfig,
        schema: CommandCollection | DictConfig,
        train_schema: CommandCollection | DictConfig,
        ignore_keywords_with_failed_value_resolution: bool = False,
        batch_size: int = 1,
        optimise_prompt: bool = True,
    ):
        super().__init__(pipeline_config, generation_config, schema, batch_size)
        self._predictions_history: dict[
            DialogueID, list[ProcessedConstraintRequest]
        ] = defaultdict(list)
        self._optimise_prompt = optimise_prompt
        self._pending_requests: dict[SlotConstraintRequest, list[DialogueID]] = {}
        self._train_schema = train_schema
        self.ignore_keywords_with_failed_value_resolution = (
            ignore_keywords_with_failed_value_resolution
        )
        self._cache: dict[ConstraintRequest, ConstrainedKeyword] = {}
        if isinstance(train_schema, DictConfig):
            self._train_schema = instantiate(train_schema)

    @property
    def optimised(self) -> bool:
        return self._optimise_prompt

    @property
    def clear_prediction_history(self) -> dict[DialogueID, dict]:
        """Return the predictions made so far. Resets the
        prediction history container."""
        serialised = {
            id_: [m.model_dump() for m in self._predictions_history[id_]]
            for id_ in self._predictions_history
        }
        self._predictions_history = defaultdict(list)
        return serialised

    def __getitem__(self, item: SlotConstraintRequest) -> ConstrainedKeyword | None:
        self._total_requests += 1
        if item in self._cache:
            self._cache_returns += 1
            return self._cache[item]
        return

    def pop_result(self, session_id: DialogueID) -> list[ProcessedConstraintRequest]:
        """Get the slot constraints for a given session."""

        results = self._processed.pop(session_id)
        self._predictions_history[session_id].extend(results)
        return results

    def get_feedback_for_session(
        self, session_id: DialogueID
    ) -> PyTODAssistantFeedback:
        """An endpoint callers can use to communicate helper feedback to the session."""
        results = self._processed.pop(session_id)
        constrained_slots = defaultdict(list)
        for r in results:
            if (correction := r.constrained_keyword.keyword) != INVALID_SLOT:
                constrained_slots[r.request.service].append(correction)
        if constrained_slots:
            return PyTODAssistantFeedback.model_validate(
                {"corrected_keywords": dict(constrained_slots)}
            )
        return PyTODAssistantFeedback.model_validate({})

    @singledispatchmethod
    def _select_template(self, request: ConstraintRequest) -> ConstraintTemplate:
        """Select a template for prompt generation according to the data type
        and whether the slot was memorised from a training domain."""
        raise NotImplementedError(f"Unsupported request type: {request}")

    @_select_template.register
    def _(self, request: SlotConstraintRequest) -> ConstraintTemplate:
        if request.categorical:
            cls = HallucinatedArgNameCatValueConstraintTemplate
            logger.debug(f"Dispatched to {cls.__name__}")
            return cls(self._schema, optimise_prompt=self._optimise_prompt)
        match request.memorisation_info:
            case None:
                cls = HallucinatedArgNameConstraintTemplate
                logger.debug(f"Dispatched to {cls.__name__}")
                return cls(self._schema, optimise_prompt=self._optimise_prompt)
            case MemorisationInfo():
                cls = MemorisedArgumentNameConstraintTemplate
                logger.debug(f"Dispatched to {cls.__name__}")
                return cls(
                    self._schema,
                    self._train_schema,
                    optimise_prompt=self._optimise_prompt,
                )
            case _:
                raise TypeError(
                    f"Unexpected value for memorisation metadata: {request.memorisation_info}"
                )

    @_select_template.register
    def _(self, request: ValueConstraintRequest) -> ConstraintTemplate:
        cls = HallucinatedBoolEnumArgValueTemplate
        logger.debug(f"Dispatched to {cls.__name__}")
        return cls(self._schema, optimise_prompt=self._optimise_prompt)

    def _queue(self, session_id: DialogueID, request: SlotConstraintRequest):
        template = self._select_template(request)
        prompt = template.get_prompt(request)
        to_queue = {"request": request, "session_id": session_id}
        to_queue.update(prompt.model_dump())
        self._processing_queue.append(KeywordConstraintRequest.model_validate(to_queue))

    def queue_request(self, session_id: DialogueID, request: SlotConstraintRequest):
        """Request the agent to constrain a slot value. Caller should first check
        if the request has already been processed and can be retrieved from the cache.
        """

        if request in self._pending_requests:
            logger.debug(
                f"{session_id}: A request similar to {str(request)} has already been queued."
            )
            self._pending_requests[request].append(session_id)
            return
        self._pending_requests[request] = []
        logger.debug(f"{session_id} Queuing request: {str(request)}")
        self._queue(session_id, request)

    def predict(self):
        """Prompt a language model to constrain slot names to valid schema slots."""
        logger.info(f"There are {len(self.queue)} datapoints to process.")
        self.queue.sort(key=lambda x: len(x.prompt))
        prompts = [el.prompt for el in self.queue]
        sessions = [el.session_id for el in self.queue]
        with self._pipeline_context():
            for e, s in zip(prompts, sessions):
                nb_tokens = len(self._pipeline.tokenizer(e)["input_ids"])
                try:
                    assert nb_tokens < 512
                except AssertionError:
                    logger.warning(f"Session {s} exceeds 512 tokens. Prompt: {e}")
            outputs = self._pipeline(
                prompts, batch_size=self._batch_size, **self._generation_config
            )
            for idx, output in enumerate(outputs):
                item = self.queue[idx]
                processed_result = ProcessedConstraintRequest.model_validate(
                    item.model_dump()
                )
                processed_result.selected_option = output["generated_text"]
                self._update_processed_results(processed_result)
                self._cache[
                    processed_result.request
                ] = processed_result.constrained_keyword
        self._pending_requests = {}

    def _update_processed_results(self, processed_result: ProcessedConstraintRequest):
        """Track the processed records for all requests, including the duplicate ones."""
        self._processed[processed_result.session_id].append(processed_result)
        if (request := processed_result.request) in self._pending_requests:
            for session_id in self._pending_requests[request]:
                self._processed[session_id].append(processed_result)
