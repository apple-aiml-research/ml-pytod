#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import json
import logging
import re
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from pydantic import BaseModel, field_validator
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from typing_extensions import Self

from pytod.command import CommandCollection
from pytod.inference.assistant import PyTODAssistant, Request
from pytod.prompting.pytod_nlu_formatters import UNK_VALUE
from pytod.pytod_types.aliases import DialogueID, SlotName
from pytod.simulation.command import SlotValue
from pytod.utils import load_json, save_json

logger = logging.getLogger(__name__)

QuestionIdx = int
UNDEFINED = "N/A"


class SupervisionRequest(Request):
    prompt: str
    dialogue_id: DialogueID
    turn_idx: int
    parser_map: dict[int, SlotName] | None = None
    gold_completion: dict[SlotName, SlotValue] | None = None
    unseen: bool | None = None

    def __hash__(self):
        return hash(f"{self.dialogue_id}-{self.turn_idx}")

    def __eq__(self, other: Self) -> bool:
        return self.dialogue_id == other.dialogue_id and self.turn_idx == other.turn_idx

    def __str__(self) -> str:
        return f"{self.dialogue_id}-{self.turn_idx}"

    @field_validator("parser_map", mode="before")
    @classmethod
    def cast_to_dict(cls, v: str | dict[int, str]) -> dict[QuestionIdx, SlotName]:
        if isinstance(v, str):
            return {int(k): v for k, v in json.loads(v).items()}
        return v


def parse_answers(prediction: str) -> dict[QuestionIdx, SlotValue]:
    # Split the string by patterns like '1)', '2)', etc.
    segments = re.split(r"(\d+\))", prediction)

    result = {}
    for i in range(1, len(segments) - 1, 2):
        index = int(segments[i][:-1])
        answer = segments[i + 1].strip()
        result[index] = answer

    return result


class ProcessedSupervisionRequest(BaseModel):
    request: SupervisionRequest
    _prediction: dict[SlotName, SlotValue] = {}

    @property
    def prediction(self) -> dict[SlotName, SlotValue] | None:
        return self._prediction or None

    @prediction.setter
    def prediction(self, pred: str):
        parser_map = self.request.parser_map
        answers = parse_answers(pred)
        for q_id, answer in answers.items():
            try:
                assert q_id in parser_map
            except AssertionError:
                logger.error(
                    f"{self.request}: Supervisor output an invalid index {q_id}. "
                    f"Parser map: {parser_map}"
                )
                continue
            if answer == UNK_VALUE and self.request.turn_idx != 0:
                logger.warning(
                    f"{self.request} Supervisor failed to parse slot {parser_map[q_id]}"
                )
            self._prediction[parser_map[q_id]] = answer


def supervisor_exact_match(
    cache: dict[SupervisionRequest, dict[SlotName, SlotValue]]
) -> float | None:
    total, correct = 0, 0
    for request, pred in cache.items():
        total += 1
        correct += int(request.gold_completion == pred)
    if total > 0:
        return (100 * correct) / total
    return None


class PolicySupervisor(PyTODAssistant):
    def __init__(
        self,
        pipeline_config: DictConfig | None,
        generation_config: DictConfig | None,
        schema: CommandCollection | DictConfig,
        batch_size: int = 1,
        ids: list[DialogueID] | None = None,
        cache_dir: str | Path | None = None,
        eval_first_turn: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            pipeline_config, generation_config, schema, batch_size=batch_size
        )
        self._cache: dict[SupervisionRequest, dict[SlotName, SlotValue]]
        if cache_dir is not None:
            logger.info(
                f"Initialising policy supervisor state from cache at {cache_dir}"
            )
            self.from_prediction_cache(cache_dir)
        self._dataset: Dataset | None = None
        self._ids = ids
        self._debug = debug
        self.first_turn_supervisor = eval_first_turn

    def __getitem__(self, item: SupervisionRequest) -> dict[SlotName, SlotValue] | None:
        self._total_requests += 1
        if item in self._cache:
            self._cache_returns += 1
            return self._cache[item]
        return

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset
        if self._ids is not None:
            self._dataset = self._dataset.filter(
                function=lambda example: example["dialogue_id"] in self._ids
            )

    def queue_request(self, session_id: DialogueID, request: Request):
        raise NotImplementedError

    def predict(self):
        if self._dataset is None:
            raise ValueError(
                "Missing prediction dataset, pass it to the agent by calling"
                "`set_dataset`."
            )
        sorted_ds = self._dataset.sort(["prompt", "completion"])
        with self._pipeline_context():
            for record, output in zip(
                sorted_ds,
                tqdm(
                    self._pipeline(
                        KeyDataset(sorted_ds, "prompt"),
                        batch_size=self._batch_size,
                        **self._generation_config,
                    )
                ),
            ):
                gold_completion = record["gold_completion"]
                unseen = record["unseen"]
                request = SupervisionRequest.model_validate(
                    {
                        "service": record["service"],
                        "prompt": record["prompt"],
                        "dialogue_id": record["dialogue_id"],
                        "turn_idx": record["turn_idx"],
                        "parser_map": record["parser_map"],
                        "gold_completion": json.loads(gold_completion)
                        if gold_completion is not None
                        else None,
                        "unseen": unseen if unseen is not None else None,
                    }
                )
                processed = ProcessedSupervisionRequest.model_validate(
                    {"request": request},
                )
                processed.prediction = output[0]["generated_text"]
                self._cache[request] = processed.prediction

    def from_prediction_cache(self, cache_dir: Path | str):
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        cache_path = cache_dir / "supervisor.json"
        predictions = load_json(cache_path)
        for id_, prediction in predictions.items():
            request = SupervisionRequest.model_validate(prediction["request"])
            self._cache[request] = prediction["prediction"]
        if (first_turn_cache := cache_dir / "supervisor_first_turn.json").exists():
            logger.info(f"Loading first turn cache from {first_turn_cache}")
            first_turn_predictions = load_json(first_turn_cache)
            for id_, prediction in first_turn_predictions.items():
                request = SupervisionRequest.model_validate(prediction["request"])
                self._cache[request] = prediction["prediction"]

    def save_predictions_cache(self, cache_dir: Path):
        cache = {}
        for request, prediction in self._cache.items():
            cache[f"{request.dialogue_id}-{request.turn_idx}"] = {
                "request": request.model_dump(),
                "prediction": prediction,
            }
        base_name_prefix = (
            "supervisor_first_turn" if self.first_turn_supervisor else "supervisor"
        )
        name = (
            f"{base_name_prefix}_debug.json"
            if self._debug
            else f"{base_name_prefix}.json"
        )
        cache_path = Path(cache_dir) / name
        save_json(cache, cache_path)

    def save_metrics(self, cache_dir: Path) -> dict[str, float]:
        base_name_prefix = (
            "metrics_first_turn" if self.first_turn_supervisor else "metrics"
        )
        metrics_name = (
            f"{base_name_prefix}_debug.json"
            if self._debug
            else f"{base_name_prefix}.json"
        )
        metrics_pth = Path(cache_dir) / metrics_name
        metrics = {
            "exact match": self.exact_match,
            "exact match (unseen)": self.unseen_api_exact_match,
            "exact match (seen)": self.seen_api_exact_match,
        }
        save_json(
            metrics,
            metrics_pth,
        )
        return metrics

    @property
    def exact_match(self) -> float | str:
        em = supervisor_exact_match(self._cache)
        if em is not None:
            return em
        return UNDEFINED

    @property
    def seen_api_exact_match(self) -> float | str:
        seen_examples = {k: v for k, v in self._cache.items() if k.unseen is False}
        if seen_examples:
            return supervisor_exact_match(seen_examples)
        return UNDEFINED

    @property
    def unseen_api_exact_match(self) -> float | str:
        unseen_examples = {k: v for k, v in self._cache.items() if k.unseen is True}
        if unseen_examples:
            return supervisor_exact_match(unseen_examples)
        return UNDEFINED
