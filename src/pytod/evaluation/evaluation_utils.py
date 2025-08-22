#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
"""Utility functions for evaluation with the official SGD evaluator."""
import collections
import glob
import json
import logging
import math
import re
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, TypedDict

import numpy as np
from omegaconf import DictConfig

from pytod.evaluation import dst_metrics
from pytod.pytod_types.aliases import (
    DialogueID,
    ServiceName,
    SGDDialogueDict,
    SGDServiceSchema,
    ShardName,
    StateDict,
)
from pytod.trainer.cli import CustomSeq2SeqTrainingArguments
from pytod.utils import (
    cast_vals_to_sorted_list,
    filter_shards,
    get_sgd_shard_name,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)

TurnIdx = int  # user turn index
"""Dict containing the inputs to DSTC8 evaluator."""
PredictionsDict = (
    dict[
        ShardName,  # eg dialogues_001.json
        dict[
            DialogueID,  # eg 1_00000
            dict[
                TurnIdx,
                dict[Literal["state", "utterance"], dict[ServiceName, StateDict]],
            ],
        ],
    ],
)
PER_FRAME_OUTPUT_FILENAME = "metrics_and_dialogues.json"
AGGREGATE_METRICS_PATTERN = r"model_\d+_metrics\.json"
ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"
AGGREGATE_METRICS_KEYS = [ALL_SERVICES, SEEN_SERVICES, UNSEEN_SERVICES]
_TRACKED_METRICS = [
    "average_goal_accuracy",
    "average_cat_accuracy",
    "average_noncat_accuracy",
    "joint_goal_accuracy",
    "joint_cat_accuracy",
    "joint_noncat_accuracy",
    "system_task_completion",
    "system_followup_task_completion",
]
# {dialogue_id}-{frame_no}-{service}
#  where frame-no is the frame index in
# zero-padded notation (eg 002)
FrameID = str
KEEP_SERVICES = [
    "Hotels_1",
    "Hotels_11",
    "Hotels_12",
    "Hotels_13",
    "Hotels_14",
    "Hotels_15",
    "Hotels_2",
    "Hotels_21",
    "Hotels_22",
    "Hotels_23",
    "Hotels_24",
    "Hotels_25",
    "Services_1",
    "Services_11",
    "Services_12",
    "Services_13",
    "Services_14",
    "Services_15",
    "Services_4",
    "Services_41",
    "Services_42",
    "Services_43",
    "Services_44",
    "Services_45",
    "Hotels_4",
    "Hotels_41",
    "Hotels_42",
    "Hotels_43",
    "Hotels_44",
    "Hotels_45",
    "Movies_1",
    "Movies_11",
    "Movies_12",
    "Movies_13",
    "Movies_14",
    "Movies_15",
    "Movies_3",
    "Movies_31",
    "Movies_32",
    "Movies_33",
    "Movies_34",
    "Movies_35",
]
DomainKey = str
# a dialogue metric output by the evaluator
# see _TRACKED_METRICS for examples
MetricKey = str


class EvaluatorInputs(TypedDict):
    eval_services: dict[ServiceName, SGDServiceSchema]
    in_domain_services: set[ServiceName]
    dataset_ref: dict[DialogueID, dict]


def get_dialogue_ids(preprocessed_refs: list[dict]) -> list[DialogueID]:
    return list({ref["dialogue_id"] for ref in preprocessed_refs})


def get_dialogue_filenames(preprocessed_refs: list[dict]) -> list[ShardName]:
    return list({ref["shard_name"] for ref in preprocessed_refs})


def get_refs_subset(
    preprocessed_refs: list[dict],
    dialogue_ids: list[DialogueID],
    shard_names: list[ShardName],
) -> list[dict]:
    if any((dialogue_ids is None, shard_names is None)):
        return preprocessed_refs
    preprocessed_refs_subset = []
    for r in preprocessed_refs:
        if r["file_name"] in shard_names and r["dialogue_id"] in dialogue_ids:
            preprocessed_refs_subset.append(r)
    assert preprocessed_refs_subset
    return preprocessed_refs_subset


def get_dataset_as_dict(file_path_patterns, decoded_only: Optional[list[str]] = None):
    """Read the DSTC8 json dialog data as dictionary with dialog ID as keys.

    Parameters
    ----------
    decoded_only
        Used for code testing with few dialogues. Should contain valid dialogue IDs.
    """
    dataset_dict = {}
    if isinstance(file_path_patterns, list):
        list_fp = file_path_patterns
    else:
        list_fp = sorted(glob.glob(file_path_patterns))
    for fp in list_fp:
        if PER_FRAME_OUTPUT_FILENAME in fp or "belief" in fp:
            continue
        logger.debug("Loading file: %s", fp)
        with open(fp, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dial_id = dial["dialogue_id"]
                    if decoded_only is not None and dial_id not in decoded_only:
                        continue
                    dataset_dict[dial_id] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
    return dataset_dict


def get_service_set(schema_path):
    """Get the set of all services present in a schema."""
    service_set = set()
    with open(schema_path, "r") as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
    return service_set


def get_in_domain_services(schema_path_1, schema_path_2):
    """Get the set of common services between two schemas."""
    return get_service_set(schema_path_1) & get_service_set(schema_path_2)


def setup_sgd_evaluator_inputs(
    ref_dir: str | Path, decoded_only: Optional[list[str]] = None
) -> EvaluatorInputs:
    """Helper function for calling evaluation from training script.

    Parameters
    ----------
    ref_dir
        The directory where the SGD-formatted dialogue files "dialogues_***.json"
        are located.
    decoded_only:
        IDs of dialogues which have been decoded.
        Used to test code with a subsample of the data.

    Returns
    -------
    A mapping containing the positional arguments of the official SGD evaluation script.
    """
    if isinstance(ref_dir, str):
        ref_dir = Path(ref_dir)
    ref_data = get_dataset_as_dict(
        str(ref_dir.joinpath("dialogues_*.json")), decoded_only=decoded_only
    )
    eval_schema_path = ref_dir.joinpath("schema.json")
    with open(eval_schema_path, "r") as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service
    # this is the schema of the SGD-X train dataset. We consider "seen" all
    # dialogues which appear in the training set. So even though the schema for
    # e.g., Homes_15 is not seen in training, for evaluation the dialogues in
    # test Homes_15 are considered seen - we saw the dialogues but not the descriptions.
    eval_variant_train_schema_path = ref_dir.parent.joinpath("train", "schema.json")
    assert (
        eval_variant_train_schema_path.exists()
    ), "Could not find the train/ subdir when setting up SGD evaluation"
    in_domain_services = get_in_domain_services(
        eval_schema_path, eval_variant_train_schema_path
    )
    logger.debug(f"In domain services: {list(in_domain_services)}")
    logger.debug(f"Evaluation schema path {eval_schema_path}")
    logger.debug(f"Evaluation services {list(eval_services.keys())}")
    return {
        "eval_services": eval_services,
        "in_domain_services": in_domain_services,
        "dataset_ref": ref_data,
    }


def setup_sgd_evaluation(
    raw_preprocessed_refs: dict[str, list[dict]],
    split: Literal["validation", "test"],
    split_ref_dir: str,
    split_template_dir: str,
    max_eval_samples: Optional[int] = None,
    max_predict_samples: Optional[int] = None,
) -> tuple[dict, dict]:
    assert split in ["validation", "test"], (
        f"Cannot setup SGD evaluation for split {split}. "
        "Valid options are 'validation' and 'test'."
    )
    logger.debug("Setting up SGD evaluation")
    files_decoded, dialogues_decoded = None, None
    if any(
        (
            max_eval_samples is not None,
            max_predict_samples is not None,
        )
    ):
        files_decoded = get_dialogue_filenames(raw_preprocessed_refs[split])
        dialogues_decoded = get_dialogue_ids(raw_preprocessed_refs[split])
    schema_path = Path(split_ref_dir).joinpath("schema.json")
    logger.debug(f"Retrieving schema from path {schema_path}")
    parser_inputs = {
        "template_dir": split_template_dir,
        "schema_path": schema_path,
        "files_to_parse": files_decoded,
        "dialogues_decoded": dialogues_decoded,
        "preprocessed_refs": get_refs_subset(
            raw_preprocessed_refs[split],
            dialogues_decoded,
            files_decoded,
        ),
    }
    evaluator_inputs = setup_sgd_evaluator_inputs(
        Path(split_ref_dir), decoded_only=dialogues_decoded
    )
    return parser_inputs, evaluator_inputs


def infer_step(output_dir: str) -> int:
    checkpoint_dir = Path(output_dir)
    if "checkpoint-" in (ckpt := checkpoint_dir.name):
        return int(ckpt.split("-")[1])
    raise ValueError(
        f"Wrong output_dir setting in training_args. Got {checkpoint_dir}."
    )


def setup_evaluator_output_dirs(
    training_args: CustomSeq2SeqTrainingArguments,
    split: Literal["test", "dev"],
    step: Optional[int] = None,
) -> tuple[Path, Path]:
    """Create the folder hierarchy for storing evaluator aggregated and
    frame-level outputs.

    This is used both during traing to save dev set model predictions and
    task-oriented eval results as well as during inference.
    """

    assert split in [
        "test",
        "dev",
    ], f"Expected split to be either 'dev' or 'test' but got {split}"
    logger.info("Setting up directories to store evaluation output")
    checkpoint_dir = Path(training_args.output_dir)
    logger.info(f"Checkpoint directory is {checkpoint_dir}")
    # checkpoint dir is suffixed with /checkpoint-[step] in inference
    if "checkpoint-" in checkpoint_dir.name:
        if step is None:
            step = infer_step(training_args.output_dir)
        input_data_version = checkpoint_dir.parent.name
        experiment_name = checkpoint_dir.parent.parent.name
    else:
        assert step is not None
        input_data_version = checkpoint_dir.name
        experiment_name = checkpoint_dir.parent.name
        try:
            assert (
                "models" in checkpoint_dir.parent.parent.name
            ), f"checkpoint dir ancestor is {checkpoint_dir.parent.parent.name}"
        except AssertionError:
            assert (
                "output" in checkpoint_dir.parent.parent.name
            ), f"checkpoint dir ancestor is {checkpoint_dir.parent.parent.name}"
    version_pattern = r"\bv\d+\.\d+(\.\d+)?\b"
    assert (
        re.search(version_pattern, input_data_version) is not None
    ), f"Input data version is {input_data_version}"
    data_variant = training_args.data_variant
    assert data_variant is not None
    logger.info(f"Inferred input data version: {input_data_version}")
    logger.info(f"Inferred experiment name: {experiment_name}")
    logger.info(f"Inferred schema variant: {data_variant}")
    if suffix := training_args.experiment_name_suffix:
        logger.info(f"Appending suffix {suffix} to experiment {experiment_name}")
        experiment_name = f"{experiment_name}_{suffix}"
    dir_hierarchy = (
        experiment_name,
        data_variant,
        split,
        input_data_version,
    )
    hyp_dir = Path(training_args.hyps_dir).joinpath(
        *dir_hierarchy, f"checkpoint-{step}"
    )
    metrics_dir = Path(training_args.metrics_dir).joinpath(*dir_hierarchy)
    if not hyp_dir.exists():
        logger.info(f"Creating hyps directory {str(hyp_dir)}")
        hyp_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_dir.exists():
        logger.info(f"Creating metrics directory {str(metrics_dir)}")
        metrics_dir.mkdir(parents=True, exist_ok=True)
    return hyp_dir, metrics_dir


def save_evaluator_outputs(
    hyp_dir: Path,
    metrics_dir: Path,
    outputs: dict[str, Any],
    step: int,
):
    """
    Save evaluator aggregated metrics, frame level output and raw model predictions.

    Parameters
    ----------
    hyp_dir, metrics_dir:
        Directories where the evaluator outputs are saved. The detailed,
        frame level output is saved in `hyp_dir` whereas the aggregated metrics
        in `metrics_dir`. `hyp_dir` is joined with `checkpoint-[step]`
        so that SGD-format dialogue files containing model predictions and
        detailed output are saved for a given checkpoint.
    outputs:
        Evaluation outputs to be saved. Each key represents the name of a file.
    step:
        Represents the training step (unit is number of weight updates) at which
        evaluation was carried out. Used to save the evaluator results in a
        subdirectory of `hyp_dir` that corresponds to the evaluated checkpoint.
    """
    if not metrics_dir.exists():
        metrics_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in outputs.items():
        dir = metrics_dir if re.match(AGGREGATE_METRICS_PATTERN, fname) else hyp_dir
        ofile = dir.joinpath(fname)
        logger.info(f"Saving {fname.split('.')[0]} at {ofile}")
        save_json(data, ofile)


def read_dial_history_shards(
    path: str, return_only: Optional[set[DialogueID]] = None
) -> dict[DialogueID, list[dict]]:
    path = Path(path)
    ffs = sorted(path.glob("dialogues*.json"))
    ffs, shards_to_dials = filter_shards(ffs, return_only=return_only)
    data = {}
    for ff in ffs:
        shard_name = ff.name
        if "metrics" in shard_name:
            continue
        dialogues = load_json(ff)
        for dial_id in dialogues:
            if shards_to_dials is None:
                data[dial_id] = dialogues[dial_id]
            else:
                if dial_id in shards_to_dials[shard_name]:
                    data[dial_id] = dialogues[dial_id]
    return data


def _write_sgd_formatted_dialogues(
    sgd_format_predictions: dict[str, list[dict]],
    name_to_fpath: dict[str, Path],
) -> None:
    """Write the sgd dialogues filled with predictions.

    Args:
        sgd_format_predictions: Dictionary of SGD dialogue file name to dialogues
            filled with model predictions
        name_to_fpath: A mapping from file name to file path
    """
    if not name_to_fpath:
        return
    logger.info("Writing SGD-formatted dialogues...")
    for fname in sgd_format_predictions:
        with open(name_to_fpath[fname], "w") as f:
            json.dump(sgd_format_predictions[fname], f, indent=2)
    logger.info("Completed parsing!")


class SGDFileMapper:
    def __init__(
        self,
        template_dir: Optional[Path | str] = None,
        hyps_dir: Optional[Path | str] = None,
        dialogues_to_map: Optional[list[str]] = None,
        corrections_info: Optional[dict[DialogueID, dict[str, str]]] = None,
    ):
        """Instantiate a parser object.

        Parameters
        -----------
        template_dir
            Path to the directory containing the blank dialogue templates
                to be filled with predictions
        hyps_dir
            Absolute Path to the directory where the hypothesis are to be written
        dialogues_to_map
            Which dialogues are expected to be copied in the output files.
        """
        self.template_dir = template_dir
        self.dialogues_to_map = dialogues_to_map
        self._hyps_dir = Path(hyps_dir) if hyps_dir is not None else None
        self._corrections_info = corrections_info or {}

    @property
    def hyps_dir(self) -> Optional[Path]:
        return self._hyps_dir

    @hyps_dir.setter
    def hyps_dir(self, value: str):
        self._hyps_dir = Path(value)

    @staticmethod
    def _load_sgd_templates(
        template_dir: Path | str,
        hyps_dir: Optional[Path] = None,
        dialogues_to_map: Optional[list[str]] = None,
        dialogues_to_exclude: Optional[list[str]] = None,
    ) -> tuple[dict[str, Path], dict[str, list[dict]]]:
        """Load blank dialogue files as templates.

        Args:
            template_dir: Path to the directory containing the blank dialogue templates
                to be filled
            hyps_dir: Optional path where the SGD-format hypotheses are to be saved.
            dialogues_to_map: Optional list of dialogue ids to parse a subset of
                dialogues
            dialogues_to_exclude: list of dialogues to skip during loading

        Returns:
            A mapping from file name to file path, and a mapping from SGD file names to
            list of SGD-formatted dialogue templates without annotations.
        """
        logger.debug("Loading SGD dialogue templates")
        files_to_map = None
        if dialogues_to_map is not None:
            files_to_map = list({get_sgd_shard_name(dial) for dial in dialogues_to_map})
        if not any((files_to_map is None, dialogues_to_map is None)):
            logger.debug(f"There are {len(dialogues_to_map)} dialogues to parse")
        pattern = re.compile(r"dialogues_[0-9]+\.json")
        sgd_files_paths = list(Path(template_dir).glob("*.json"))
        sgd_files_paths = sorted(
            [p for p in sgd_files_paths if pattern.match(p.name)],
            key=lambda p: int((p.name.split("_")[1]).split(".")[0]),
        )
        sgd_dialogue_templates = defaultdict(list)
        name_to_fpath = {}
        for fpath in sgd_files_paths:
            fname = fpath.name
            if files_to_map is not None and fname not in files_to_map:
                logger.debug(
                    f"Skipping {fpath.name} as it is not amongst files to parse. "
                    "This is expected if you are testing with few samples."
                )
                continue
            with open(fpath, "r") as f:
                raw_dialogues = json.load(f)
            # exclude files/dialogues to support testing/debugging
            if dialogues_to_map is not None:
                logger.debug(
                    "Some dialogues will be skipped during template loading as only a"
                    " subset have been decoded"
                )
                raw_dialogues = [
                    dial
                    for dial in raw_dialogues
                    if dial["dialogue_id"] in dialogues_to_map
                ]
            if dialogues_to_exclude is not None:
                raw_dialogues = [
                    dial
                    for dial in raw_dialogues
                    if dial["dialogue_id"] not in dialogues_to_exclude
                ]
            sgd_dialogue_templates[fname] = raw_dialogues
            if hyps_dir is not None:
                name_to_fpath[fname] = hyps_dir / fname

        return name_to_fpath, dict(sgd_dialogue_templates)

    def _copy_predictions_to_sgd_format(
        self, shard: str, raw_sgd_dialogues: list[dict], predictions: PredictionsDict
    ) -> None:
        """Copies predictions in SGD frames.

        If predictions are not found, dialogues are ignored and dialogues removed from
        the input list and references.

        Args:
            shard: Name of the dialogue file (eg dialogues_001.json)
            raw_sgd_dialogues: List of the blank sgd dialogue templates
            predictions: dialogue state predictions
        """

        for dial in raw_sgd_dialogues:
            dial_id: DialogueID = dial["dialogue_id"]
            # user+sys if only user turn predictions
            num_turn_predictions = len(predictions[shard].get(dial_id, [])) * 2
            assert len(dial["turns"]) == num_turn_predictions
            for turn_idx, turn in enumerate(dial["turns"]):
                if turn["speaker"] == "SYSTEM":
                    continue
                this_turn_usr_utt = predictions[shard][dial_id][turn_idx]["utterance"]
                try:
                    assert this_turn_usr_utt.lower() == turn["utterance"].lower()
                except AssertionError:
                    assert dial_id in self._corrections_info
                this_turn_pred = predictions[shard][dial_id][turn_idx]["state"]
                try:
                    assert len(turn["frames"]) <= len(this_turn_pred)
                except AssertionError:
                    print(dial_id)
                    raise AssertionError
                for frame_idx, frame in enumerate(turn["frames"]):
                    service = frame["service"]
                    try:
                        this_service_pred = this_turn_pred[service]
                        slot_values_prediction = this_service_pred["slot_values"]
                        requested_slots = this_service_pred["requested_slots"]
                        active_intent = this_service_pred["active_intent"]
                    except KeyError:
                        logger.warning(
                            f"{dial_id}-{turn_idx} Could not find predictions for"
                            f" service {service} in dialogue"
                        )
                        raise KeyError(
                            f"Could not find {service} state among predictions"
                        )
                    frame["state"]["slot_values"] = slot_values_prediction
                    frame["state"]["requested_slots"] = requested_slots
                    frame["state"]["active_intent"] = active_intent

    def convert_to_sgd_format(
        self,
        predictions: PredictionsDict,
        write_to_disk: bool = False,
    ) -> dict[ShardName, list[SGDDialogueDict]]:
        """Replace the annotations of `sgd_dialogue_templates` with model predictions.

        This method requires template_dir attribute to be set on the Parser instance.

        Parameters
        ----------
            predictions: Outputs of the executed programs.
            write_to_disk: Whether to write the SGD formatted dialogues to disk.


        Returns
        -------
            Dictionary of SGD dialogue file name to dialogues filled with model
            predictions
        Notes
        -----
            Each string in the predictions is mapped to slot-value pairs using
            preprocessed_refs by the _map_values_to_slots method.


        """
        if self.template_dir is None:
            raise RuntimeError("template_dir is required for SGD parsing")
        name_to_fpath, sgd_dialogue_templates = self._load_sgd_templates(
            template_dir=self.template_dir,
            dialogues_to_map=self.dialogues_to_map,
            hyps_dir=self._hyps_dir,
        )
        # a mapping from SGD file names to list of SGD-formatted dialogue templates
        # without annotations
        sgd_format_predictions = deepcopy(sgd_dialogue_templates)
        logger.debug("Mapping predictions to SGD format...")
        for fname, raw_dialogues in sgd_format_predictions.items():
            logger.debug(f"Converting file {fname}")
            assert len(raw_dialogues) == len(predictions[fname])
            self._copy_predictions_to_sgd_format(fname, raw_dialogues, predictions)

        if write_to_disk:
            _write_sgd_formatted_dialogues(sgd_format_predictions, name_to_fpath)

        return sgd_format_predictions


def flatten_metrics_dict(sgd_metrics_dict: dict) -> dict[str, float]:
    """
    Turn SGD evaluator output into a mapping from key to values so that it can
    be logged by services such as Tensorboard or ``wandb``.
    """

    def update_with_metrics(
        flattened_dict: dict[str, float],
        metric_key: str,
        sgd_metrics_dict: dict[str, dict[str, float]],
    ):
        for metric_name, value in sgd_metrics_dict[metric_key].items():
            if metric_name in _TRACKED_METRICS:
                flattened_dict[f"{metric_key}/{metric_name}"] = value
            if any(
                metric_name.startswith(m) for m in dst_metrics.TASK_COMPLETION_METRICS
            ):
                flattened_dict[f"{metric_key}/{metric_name.replace('/', '_')}"] = value

    service_categories = [ALL_SERVICES, UNSEEN_SERVICES, SEEN_SERVICES]
    flattened_dict = {}
    for service_type in service_categories:
        if service_type not in sgd_metrics_dict:
            logger.debug(
                f"No {service_type} were scored. This is expected if you are testing"
                " code with few samples."
            )
            continue
        this_service_type_metrics = sgd_metrics_dict[service_type]
        for metric_name, value in this_service_type_metrics.items():
            if metric_name in _TRACKED_METRICS:
                flattened_dict[f"{service_type}/{metric_name}"] = value
            if any(
                metric_name.startswith(m) for m in dst_metrics.TASK_COMPLETION_METRICS
            ):
                flattened_dict[
                    f"{service_type}/{metric_name.replace('/', '_')}"
                ] = value
    # capture domain-level metrics
    for metric_key in sgd_metrics_dict:
        # add domain level metrics
        if metric_key not in service_categories and "_" not in metric_key:
            update_with_metrics(flattened_dict, metric_key, sgd_metrics_dict)
        # keep service level metrics
        if metric_key in KEEP_SERVICES:
            update_with_metrics(flattened_dict, metric_key, sgd_metrics_dict)
    return flattened_dict


def filter_keys(all_metric_aggregate: dict[DomainKey, dict[MetricKey, float | str]]):
    """Some tasks are only follow-up tasks. To highlight this, we remove the
    system task completion metrics that has an identical value."""
    to_remove = defaultdict(list)
    for domain_key, metrics in all_metric_aggregate.items():
        for key in metrics:
            if key.startswith(dst_metrics.FOLLOWUP_TASK_COMPLETION):
                followup_value = metrics[key]
                no_followup_key = key.replace("_followup", "")
                if no_followup_key in metrics and math.isclose(
                    metrics[no_followup_key], followup_value
                ):
                    to_remove[domain_key].append(no_followup_key)
    for domain_key, keys in to_remove.items():
        for key in keys:
            all_metric_aggregate[domain_key].pop(key)


def get_metrics(
    dataset_ref,
    dataset_hyp,
    service_schemas,
    in_domain_services,
    use_fuzzy_match=True,
    joint_acc_across_turn=False,
):
    """Calculate the DSTC8 metrics.
    Args:
      joint_acc_across_turn:
      use_fuzzy_match:
      dataset_ref: The ground truth dataset represented as a dict mapping dialogue
        id to the corresponding dialogue.
      dataset_hyp: The predictions in the same format as `dataset_ref`.
      service_schemas: A dict mapping service name to the schema for the service.
      in_domain_services: The set of services which are present in the training
        set.
    Returns:
      A dict mapping a metric collection name to a dict containing the values
      for various metrics. Each metric collection aggregates the metrics across
      a specific set of frames in the dialogues.
    """
    # Metrics can be aggregated in various ways, eg over all dialogues, only for
    # dialogues containing unseen services or for dialogues corresponding to a
    # single service. This aggregation is done through metric_collections, which
    # is a dict mapping a collection name to a dict, which maps a metric to a list
    # of values for that metric. Each value in this list is the value taken by
    # the metric on a frame.
    metric_collections = collections.defaultdict(lambda: collections.defaultdict(list))

    # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
    assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))
    logger.debug(
        "len(dataset_hyp)=%d, len(dataset_ref)=%d", len(dataset_hyp), len(dataset_ref)
    )

    # Store metrics for every frame for debugging.
    per_frame_metric = {}
    for dial_id, dial_hyp in dataset_hyp.items():
        dial_ref = dataset_ref[dial_id]

        if set(dial_ref["services"]) != set(dial_hyp["services"]):
            raise ValueError(
                "Set of services present in ground truth and predictions don't match "
                "for dialogue with id {}".format(dial_id)
            )
        joint_metrics = [
            dst_metrics.JOINT_GOAL_ACCURACY,
            dst_metrics.JOINT_CAT_ACCURACY,
            dst_metrics.JOINT_NONCAT_ACCURACY,
        ]
        for turn_id, (turn_ref, turn_hyp) in enumerate(
            zip(dial_ref["turns"], dial_hyp["turns"])
        ):
            metric_collections_per_turn = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 1.0)
            )
            if turn_ref["speaker"] != turn_hyp["speaker"]:
                raise ValueError(
                    "Speakers don't match in dialogue with id {}".format(dial_id)
                )

            # Skip system turns because metrics are only computed for user turns.
            if turn_ref["speaker"] != "USER":
                continue

            if turn_ref["utterance"] != turn_hyp["utterance"]:
                logger.info("Ref utt: %s", turn_ref["utterance"])
                logger.info("Hyp utt: %s", turn_hyp["utterance"])
                raise ValueError(
                    "Utterances don't match for dialogue with id {}".format(dial_id)
                )

            hyp_frames_by_service = {
                frame["service"]: frame for frame in turn_hyp["frames"]
            }

            # Calculate metrics for each frame in each user turn.
            for frame_ref in turn_ref["frames"]:
                service_name = frame_ref["service"]
                if service_name not in hyp_frames_by_service:
                    raise ValueError(
                        "Frame for service {} not found in dialogue with id {}".format(
                            service_name, dial_id
                        )
                    )
                service = service_schemas[service_name]
                frame_hyp = hyp_frames_by_service[service_name]

                active_intent_acc = dst_metrics.get_active_intent_accuracy(
                    frame_ref, frame_hyp
                )
                slot_tagging_f1_scores = dst_metrics.get_slot_tagging_f1(
                    frame_ref, frame_hyp, turn_ref["utterance"], service
                )
                requested_slots_f1_scores = dst_metrics.get_requested_slots_f1(
                    frame_ref, frame_hyp
                )
                goal_accuracy_dict = dst_metrics.get_average_and_joint_goal_accuracy(
                    frame_ref, frame_hyp, service, use_fuzzy_match
                )
                task_completion_dict = dst_metrics.get_system_task_completion(
                    frame_ref, frame_hyp, service, use_fuzzy_match
                )
                frame_metric = {
                    dst_metrics.ACTIVE_INTENT_ACCURACY: active_intent_acc,
                    dst_metrics.REQUESTED_SLOTS_F1: requested_slots_f1_scores.f1,
                    dst_metrics.REQUESTED_SLOTS_PRECISION: requested_slots_f1_scores.precision,  # noqa: E501
                    dst_metrics.REQUESTED_SLOTS_RECALL: requested_slots_f1_scores.recall,
                }
                if slot_tagging_f1_scores is not None:
                    frame_metric[
                        dst_metrics.SLOT_TAGGING_F1
                    ] = slot_tagging_f1_scores.f1
                    frame_metric[
                        dst_metrics.SLOT_TAGGING_PRECISION
                    ] = slot_tagging_f1_scores.precision
                    frame_metric[
                        dst_metrics.SLOT_TAGGING_RECALL
                    ] = slot_tagging_f1_scores.recall
                frame_metric.update(goal_accuracy_dict)
                frame_metric.update(task_completion_dict)

                frame_id = "{:s}-{:03d}-{:s}".format(
                    dial_id, turn_id, frame_hyp["service"]
                )
                per_frame_metric[frame_id] = frame_metric
                # Add the frame-level metric result back to dialogues.
                frame_hyp["metrics"] = frame_metric

                # Get the domain name of the service.
                domain_name = frame_hyp["service"].split("_")[0]
                domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
                if frame_hyp["service"] in in_domain_services:
                    domain_keys.append(SEEN_SERVICES)
                else:
                    domain_keys.append(UNSEEN_SERVICES)
                for domain_key in domain_keys:
                    for metric_key, metric_value in frame_metric.items():
                        if metric_value != dst_metrics.NAN_VAL:
                            if joint_acc_across_turn and metric_key in joint_metrics:
                                metric_collections_per_turn[domain_key][
                                    metric_key
                                ] *= metric_value
                            else:
                                if metric_key in dst_metrics.TASK_COMPLETION_METRICS:
                                    if domain_key not in AGGREGATE_METRICS_KEYS:
                                        active_intent = frame_ref["state"][
                                            "active_intent"
                                        ]
                                        assert active_intent != "NONE"
                                        metric_key = f"{metric_key}/{active_intent}"
                                metric_collections[domain_key][metric_key].append(
                                    metric_value
                                )
            if joint_acc_across_turn:
                # Conduct multiwoz style evaluation that computes joint goal accuracy
                # across all the slot values of all the domains for each turn.
                for domain_key in metric_collections_per_turn:
                    for metric_key, metric_value in metric_collections_per_turn[
                        domain_key
                    ].items():
                        metric_collections[domain_key][metric_key].append(metric_value)
    all_metric_aggregate = {}
    for domain_key, domain_metric_vals in metric_collections.items():
        domain_metric_aggregate = {}
        for metric_key, value_list in domain_metric_vals.items():
            if value_list:
                # Metrics are macro-averaged across all frames.
                domain_metric_aggregate[metric_key] = float(np.mean(value_list))
            else:
                domain_metric_aggregate[metric_key] = dst_metrics.NAN_VAL
        all_metric_aggregate[domain_key] = domain_metric_aggregate
    filter_keys(all_metric_aggregate)
    return all_metric_aggregate, per_frame_metric


class SGDMetrics(NamedTuple):
    """Container for DSTC8 evaluator output metrics.

    Parameters
    ----------
    per_frame_metrics
        Frame level metrics.
    metrics_for_logging
        Aggregated metrics. The following keys are defined.

            #ALL_SERVICES/* | #SEEN_SERVICES/* | #UNSEEN_SERVICES/* | DOMAIN/* | KEEP_SERVICE/*

             where * : **_goal_accuracy | **_cat_accuracy | **_noncat_accuracy
             where ** : average | joint
             DOMAIN = service.split("_")[0]
             KEEP_SERVICE is one of the services above

        We keep the service-level metrics for services where multiple variants of the
        service schema are defined (members of KEEP_SERVICE) to be able to analyse
        the service-level performance.
    service_metrics:
        Same information as in `metrics_for_logging` where keys are:

            #ALL_SERVICES | #SEEN_SERVICES | #UNSEEN_SERVICES | DOMAIN | KEEP_SERVICE
    """

    per_frame_metrics: dict[FrameID, dict[MetricKey, float | str]]
    metrics_for_logging: dict[str, float]
    service_metrics: dict[DomainKey, dict[MetricKey, float]]

    def collect_execution_errors(self) -> dict[ServiceName, list[DialogueID]] | None:
        frame_metrics = self.per_frame_metrics
        errors = defaultdict(set)
        for key in frame_metrics:
            dial_id, _, service = key.split("-")
            if frame_metrics[key]["joint_goal_accuracy"] != 1.0:
                errors[service].add(dial_id)

        return (
            cast_vals_to_sorted_list(
                errors, sort_by=lambda x: tuple(int(e) for e in x.split("_"))
            )
            or None
        )


def compute_dst_metrics(
    evaluator_inputs: EvaluatorInputs,
    predictions: dict[ShardName, list[SGDDialogueDict]],
) -> SGDMetrics:
    """Run the DSTC8 evaluator on predictions."""
    dataset_hyp = {dial["dialogue_id"]: dial for dial in chain(*predictions.values())}
    all_metrics_aggregate, per_frame = get_metrics(
        dataset_ref=evaluator_inputs["dataset_ref"],
        dataset_hyp=dataset_hyp,
        service_schemas=evaluator_inputs["eval_services"],
        in_domain_services=evaluator_inputs["in_domain_services"],
    )
    return SGDMetrics(
        metrics_for_logging=flatten_metrics_dict(all_metrics_aggregate),
        per_frame_metrics=per_frame,
        service_metrics=all_metrics_aggregate,
    )