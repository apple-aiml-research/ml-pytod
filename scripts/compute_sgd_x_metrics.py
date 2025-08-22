#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from __future__ import annotations

import json
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Literal

import click
import numpy as np

from pytod.evaluation.dst_metrics import NAN_VAL
from pytod.evaluation.evaluation_utils import get_in_domain_services
from pytod.utils import (
    aggregate_values,
    append_to_values,
    default_to_regular,
    load_json,
    nested_defaultdict,
)

METRICS_FILE = "metrics_and_dialogues.json"
# METRICS_FILE = "dialogues_and_metrics.json"

_SPLIT = "test"
# _SPLIT = ''
_CHECKPOINT_PREFIX = "checkpoint"
_CHECKPOINT_SEP = "-"

TRACKED_METRICS = [
    "joint_goal_accuracy",
    "joint_cat_accuracy",
    "joint_noncat_accuracy",
]
logger = logging.getLogger(__name__)


def validate_split(ctx, param, split: str):
    assert split in ["test", "dev"]
    return split


def get_service_set(schema_path):
    """Get the set of all services present in a schema."""
    service_set = set()
    with open(schema_path, "r") as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
    return service_set


def check_processing(
    scores: dict[str, dict[str, dict[str, list[list[float]]]]],
    scores_reduced: dict[str, dict[str, list[list[list[float]]]]],
    schema_variants: list[str],
    model: str,
    split: str,
):
    for var_idx, variant in enumerate(schema_variants):
        for model_step in range(len(scores[model][variant][split])):
            assert math.isclose(
                sum(scores_reduced[model][split][model_step][var_idx]),
                sum(scores[model][variant][split][model_step]),
            )


def get_metric_sensitivity(scores: np.ndarray) -> float:
    scores = scores.T
    n_schemas = scores.shape[1]
    assert n_schemas == 5
    mean = np.mean(scores, axis=1, keepdims=True)
    std = np.sqrt(np.sum((scores - mean) ** 2, axis=1, keepdims=True) / (n_schemas - 1))
    return np.nanmean(std / mean)


def remove_nan_val(lst: list) -> list:
    """Remove NA from lists of scores where metrics are not applicable to allow averaging."""
    logger.warning(
        "Found NA in scores array. This may be expected depending on what metric you evaluate"
    )
    return [v for v in lst if v != NAN_VAL]


def filter_paths(paths: list[Path], step: str | None) -> list[Path]:
    if step is None:
        return paths
    filtered = []
    for p in paths:
        if step in p.name:
            filtered.append(p)
    return filtered


@click.command()
@click.option(
    "-vars",
    "--variants",
    "schema_variants",
    default=("v1", "v2", "v3", "v4", "v5"),
    multiple=True,
    help="Which variants will be used for computing SGD-X metrics",
)
@click.option(
    "-s",
    "--split",
    "split",
    default="test",
    help="Split to evaluate on",
    callback=validate_split,
)
@click.option(
    "-ver",
    "--version",
    "version",
    default="",
    type=str,
    required=False,
    help="The data version on which the model was trained on. Should be in the format version_*.",
)
@click.option(
    "-h",
    "--hyps_source_dir",
    "hyps_source_dir",
    default="hyps",
    type=str,
    help="Absolute to the path where the hypothesis for the models specified with -m/--models "
    "option are located",
)
@click.option(
    "-mod",
    "--models",
    "models",
    multiple=True,
    required=True,
    help="Names of the experiments for which SGD-X metrics are to be computed. This should be "
    "a subset of the names of the directories listed under -h/--hyps_source_dir option.",
)
@click.option(
    "-step",
    "--eval_step",
    "eval_step",
    multiple=True,
    default=None,
    type=str,
    help="If multiple checkpoints are available in the model directory, specify which checkpoints"
    "should be evaluated",
)
@click.option(
    "-m",
    "--metric",
    "metric",
    default="joint_goal_accuracy",
    type=str,
    help="Performance measure for which the SGD-X metrics should be evaluated. "
    "This should be a valid metric in the SGD evaluation output "
    "(e.g., joint_goal_accuracy)",
)
@click.option(
    "-a",
    "--average",
    "average_across_models",
    is_flag=True,
    default=False,
)
@click.option(
    "-o",
    "--original",
    "original",
    is_flag=True,
    default=False,
)
@click.option(
    "-sgd",
    "--sgd_dir",
    "sgd_dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
def main(
    schema_variants: tuple[str],
    split: Literal["test", "dev"],
    hyps_source_dir: str,
    version: str,
    models: tuple[str],
    eval_step: tuple[str] | None,
    metric: str,
    average_across_models: bool,
    original: bool,
    sgd_dir: Path,
):
    if original:
        schema_variants = ("original",)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    # TODO: SAVE SGD_X METRICS IN .JSON FORMAT IN APPROPRIATE LOCATION
    assert isinstance(schema_variants, tuple)
    schema_variants = list(schema_variants)
    frame_metric_paths = nested_defaultdict(list, depth=3)
    if eval_step:
        assert len(eval_step) == len(models), "Specify a step for each model"
    else:
        eval_step = tuple([None] * len(models))
    for step, model in zip(eval_step, models):
        for variant in schema_variants:
            this_model_schema_variant_paths = list(
                Path(".")
                .resolve()
                .joinpath(hyps_source_dir, model, variant, split, version)
                .glob(f"{_CHECKPOINT_PREFIX}*")
            )
            this_model_schema_variant_paths = filter_paths(
                this_model_schema_variant_paths, step
            )
            this_model_schema_variant_paths = sorted(
                this_model_schema_variant_paths,
                key=lambda pth: int(pth.name.split(f"{_CHECKPOINT_PREFIX}")[1]),
            )
            logger.info(
                f"Paths for model {model}, schema variant {variant}: "
                f"{this_model_schema_variant_paths}"
            )
            frame_metric_paths[model][variant][split].extend(
                [p.joinpath(f"{METRICS_FILE}") for p in this_model_schema_variant_paths]
            )

    frame_metrics = nested_defaultdict(list, depth=3)
    for model in models:
        for variant in schema_variants:
            frame_metrics[model][variant][split] = [
                load_json(pth) for pth in frame_metric_paths[model][variant][split]
            ]

    # Metric to use
    orig_train_schema_path = sgd_dir.absolute().joinpath("train", "schema.json")
    orig_eval_schema_path = sgd_dir.absolute().joinpath(f"{split}", "schema.json")
    in_domain_services = get_in_domain_services(
        orig_train_schema_path, orig_eval_schema_path
    )
    # Retrieve scores for all models, schema variant and optimization step
    # given a split and input data version
    all_scores = nested_defaultdict(list, depth=3)
    seen_scores = nested_defaultdict(list, depth=3)
    unseen_scores = nested_defaultdict(list, depth=3)
    all_scores_reduced = nested_defaultdict(list, depth=2)
    seen_scores_reduced = nested_defaultdict(list, depth=2)
    unseen_scores_reduced = nested_defaultdict(list, depth=2)
    average_across_variants = nested_defaultdict(list, depth=2)
    for model in models:
        (
            all_scores_across_variants,
            seen_scores_across_variants,
            unseen_scores_across_variants,
        ) = ([], [], [])
        for variant in schema_variants:
            for step_idx, this_step_frame_metrics in enumerate(
                frame_metrics[model][variant][split]
            ):
                (
                    this_step_idx_all_scores,
                    this_step_idx_seen_scores,
                    this_step_idx_unseen_scores,
                ) = ([], [], [])
                for frame_id, metric_vals in this_step_frame_metrics.items():
                    metric_value = metric_vals[metric]
                    if metric_value == NAN_VAL:
                        continue
                    this_step_idx_all_scores.append(metric_value)
                    service_name = frame_id.split("-")[-1]
                    if not original:
                        service_name = service_name[:-1]
                    if service_name in in_domain_services:
                        this_step_idx_seen_scores.append(metric_value)
                    else:
                        this_step_idx_unseen_scores.append(metric_value)
                assert len(this_step_idx_unseen_scores) + len(
                    this_step_idx_seen_scores
                ) == len(this_step_idx_all_scores)
                all_scores[model][variant][split].append(this_step_idx_all_scores)
                seen_scores[model][variant][split].append(this_step_idx_seen_scores)
                unseen_scores[model][variant][split].append(this_step_idx_unseen_scores)
            if not all_scores_across_variants:
                for model_step in range(len(all_scores[model][variant][split])):
                    all_scores_across_variants.append(
                        [remove_nan_val(all_scores[model][variant][split][model_step])]
                    )
            else:
                for model_step in range(len(all_scores[model][variant][split])):
                    all_scores_across_variants[model_step].append(
                        remove_nan_val(all_scores[model][variant][split][model_step])
                    )
            if not seen_scores_across_variants:
                for model_step in range(len(seen_scores[model][variant][split])):
                    seen_scores_across_variants.append(
                        [remove_nan_val(seen_scores[model][variant][split][model_step])]
                    )
            else:
                for model_step in range(len(seen_scores[model][variant][split])):
                    seen_scores_across_variants[model_step].append(
                        remove_nan_val(seen_scores[model][variant][split][model_step])
                    )
            if not unseen_scores_across_variants:
                for model_step in range(len(unseen_scores[model][variant][split])):
                    unseen_scores_across_variants.append(
                        [
                            remove_nan_val(
                                unseen_scores[model][variant][split][model_step]
                            )
                        ]
                    )
            else:
                for model_step in range(len(unseen_scores[model][variant][split])):
                    unseen_scores_across_variants[model_step].append(
                        remove_nan_val(unseen_scores[model][variant][split][model_step])
                    )
        logger.info(f"Model {model}")
        variant_aggregated_scores = deepcopy(all_scores[model])
        aggregate_values(variant_aggregated_scores, "mean", reduce=False)
        logger.info("Variant scores")
        variant_aggregated_scores = default_to_regular(variant_aggregated_scores)
        logger.info(variant_aggregated_scores)
        if average_across_models:
            append_to_values(
                average_across_variants, deepcopy(variant_aggregated_scores)
            )
        all_scores_reduced[model][split] = all_scores_across_variants
        seen_scores_reduced[model][split] = seen_scores_across_variants
        unseen_scores_reduced[model][split] = unseen_scores_across_variants
        check_processing(all_scores, all_scores_reduced, schema_variants, model, split)
        check_processing(
            seen_scores, seen_scores_reduced, schema_variants, model, split
        )
        check_processing(
            unseen_scores, unseen_scores_reduced, schema_variants, model, split
        )
    if average_across_models:
        aggregate_values(average_across_variants, "mean")
        logger.info(f"Average across variants for models {models}")
        logger.info(default_to_regular(average_across_variants))
    # convert reduced scores to 2-D tensor containing metrics for
    # all variants for each model optimisation step
    all_scores_arrays = nested_defaultdict(list, depth=2)
    seen_scores_arrays = nested_defaultdict(list, depth=2)
    unseen_scores_arrays = nested_defaultdict(list, depth=2)
    for model in models:
        all_scores_arrays[model][split] = [
            np.asarray(all_variant_scores)
            for all_variant_scores in all_scores_reduced[model][split]
        ]
        seen_scores_arrays[model][split] = [
            np.asarray(all_variants_seen_scores)
            for all_variants_seen_scores in seen_scores_reduced[model][split]
        ]
        unseen_scores_arrays[model][split] = [
            np.asarray(all_variants_unseen_scores)
            for all_variants_unseen_scores in unseen_scores_reduced[model][split]
        ]
    # calculate JGA 1-5
    all_jga_avg = nested_defaultdict(list, depth=2)
    seen_jga_avg = nested_defaultdict(list, depth=2)
    unseen_jga_avg = nested_defaultdict(list, depth=2)
    logger.info(f"Reporting metric: {metric}")
    jga_across_models = nested_defaultdict(list, depth=2)
    for model in models:
        logger.info(f"Model {model}")
        all_jga_avg[model][split] = [
            np.mean(arr) for arr in all_scores_arrays[model][split]
        ]
        seen_jga_avg[model][split] = [
            np.mean(arr) for arr in seen_scores_arrays[model][split]
        ]
        unseen_jga_avg[model][split] = [
            np.mean(arr) for arr in unseen_scores_arrays[model][split]
        ]
        logger.info(
            f"Average JGA for schema variants {schema_variants} on all services. "
            f"Model {model}, split {split}.{all_jga_avg[model][split]}",
        )
        logger.info(
            f"Average JGA for schema variants {schema_variants} on seen services. "
            f"Model {model}, split {split}. {seen_jga_avg[model][split]}",
        )
        logger.info(
            f"Average JGA for schema variants {schema_variants} on unseen services. "
            f"Model {model}, split {split}. {unseen_jga_avg[model][split]}",
        )
        if average_across_models:
            append_to_values(
                jga_across_models,
                {
                    "seen": seen_jga_avg[model][split],
                    "unseen": unseen_jga_avg[model][split],
                    "all": all_jga_avg[model][split],
                },
            )
    if average_across_models:
        logger.info("Average JGA across models")
        aggregate_values(jga_across_models, "mean")
        logger.info(default_to_regular(jga_across_models))

    if original:
        sys.exit()
    # calculate SS
    ss_across_models = nested_defaultdict(list, depth=2)
    all_ss = nested_defaultdict(list, depth=2)
    seen_ss = nested_defaultdict(list, depth=2)
    unseen_ss = nested_defaultdict(list, depth=2)

    for model in models:
        all_ss[model][split] = [
            get_metric_sensitivity(arr) for arr in all_scores_arrays[model][split]
        ]
        seen_ss[model][split] = [
            get_metric_sensitivity(arr) for arr in seen_scores_arrays[model][split]
        ]
        unseen_ss[model][split] = [
            get_metric_sensitivity(arr) for arr in unseen_scores_arrays[model][split]
        ]
        logger.info(
            f"Schema sensitivity for schema variants {schema_variants} on all services. "
            f"Model {model}, split {split}. {all_ss[model][split]}",
        )
        logger.info(
            f"Schema sensitivity for schema variants {schema_variants} on seen services. "
            f"Model {model}, split {split}. {seen_ss[model][split]}",
        )
        logger.info(
            f"Schema sensitivity for schema variants {schema_variants} on unseen services. "
            f"Model {model}, split {split}. {unseen_ss[model][split]}"
        )
        if average_across_models:
            append_to_values(
                ss_across_models,
                {
                    "seen": seen_ss[model][split],
                    "unseen": unseen_ss[model][split],
                    "all": all_ss[model][split],
                },
            )
    if average_across_models:
        logger.info("Average SS across models")
        aggregate_values(ss_across_models, "mean")
        logger.info(default_to_regular(ss_across_models))


if __name__ == "__main__":
    main()
