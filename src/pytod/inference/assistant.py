#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import abc
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel

from pytod.command import CommandCollection
from pytod.inference.pipelines import CustomText2TextGenerationPipeline
from pytod.pytod_types.aliases import DialogueID, ServiceName, SlotName

logger = logging.getLogger(__name__)


class Request(BaseModel, frozen=True):
    service: ServiceName

    def __hash__(self):
        raise NotImplementedError("Subclasses must implement __hash__")

    def __eq__(self, other):
        raise NotImplementedError("Subclasses must implement __eq__")


class PendingRequest(BaseModel):
    prompt: str
    session_id: DialogueID
    request: Request


class ProcessedRequest(PendingRequest):
    request: PendingRequest


RequestT = TypeVar("RequestT", bound=Request)
PendingRequestT = TypeVar("PendingRequestT", bound=PendingRequest)
ProcessedRequestT = TypeVar("ProcessedRequestT", bound=ProcessedRequest)


class PyTODAssistant(abc.ABC):
    """Base class for an agent that the PyTOD dialogue
    manager (NeuralDialogueSession) can defer to when
    receiving feedback from the parser about incorrect
    program generation."""

    def __init__(
        self,
        pipeline_config: DictConfig | None,
        generation_config: DictConfig | None,
        schema: CommandCollection | DictConfig,
        batch_size: int = 1,
    ):
        self._pipeline_config = pipeline_config
        self._initialised = False
        self._init_schema(schema)
        self._processing_queue: list[PendingRequestT] = []
        self._processed: dict[DialogueID, list[ProcessedRequestT]] = defaultdict(list)
        self._cache: dict[RequestT, Any] = {}
        self._batch_size = batch_size
        self._total_requests = 0
        self._cache_returns = 0
        if generation_config is not None:
            self._generation_config = dict(generation_config)

    def _init_schema(self, schema: CommandCollection | DictConfig):
        self._schema = schema
        if isinstance(schema, DictConfig):
            self._schema = instantiate(schema)

    def _initialise(self):
        logger.info("Initialising slot constraint pipeline")
        torch.cuda.empty_cache()
        self._pipeline: CustomText2TextGenerationPipeline = instantiate(
            self._pipeline_config, _recursive_=False
        )

    def _teardown(self):
        """Remove the constraint model from the GPU."""
        logger.info("Tearing down pipeline")
        self._pipeline.model.to("cpu")
        delattr(self._pipeline, "model")
        delattr(self, "_pipeline")
        torch.cuda.empty_cache()
        self._processing_queue = []

    @contextmanager
    def _pipeline_context(self):
        """Load a text2text generation pipeline
        wrapping the assistant model. Release
        the resources if an error occurs or
        prediction is complete.
        """
        self._initialise()
        try:
            yield
        finally:
            self._teardown()

    @property
    def queue(self) -> list[PendingRequest]:
        return self._processing_queue

    @abc.abstractmethod
    def queue_request(self, session_id: DialogueID, request: Request):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError


class PyTODAssistantFeedback(BaseModel):
    corrected_keywords: dict[ServiceName, list[SlotName]] | None = None
