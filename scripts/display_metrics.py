#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
import logging
import math
from importlib import resources
from typing import Literal

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from pytod.utils import load_json

logger = logging.getLogger(__name__)

DISPLAYED_METRICS = [
    "active_intent_accuracy",
    "joint_goal_accuracy",
    "joint_cat_accuracy",
    "joint_noncat_accuracy",
    "system_task_completion",
    "system_followup_task_completion",
]
AggregatedMetricKey = Literal["#ALL_SERVICES", "#SEEN_SERVICES", "#UNSEEN_SERVICES"]
# #ALL_SERVICES | #SEEN_SERVICES | #UNSEEN_SERVICES
#  OR DOMAIN_NAME | SERVICE_NAME
MetricKey = str
ServiceDomainKey = str
MetricName = str


def get_aggregated_metrics(
    metrics: dict[MetricKey, dict[MetricName, float]]
) -> dict[AggregatedMetricKey, dict[MetricName, float]]:
    """Extract metrics aggregated across services from the metrics file."""
    aggregated_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith("#ALL_SERVICES")
        or key.startswith("#SEEN_SERVICES")
        or key.startswith("#UNSEEN_SERVICES")
    }
    filtered_metrics = get_displayed_metrics(aggregated_metrics)
    return filtered_metrics


def is_present(
    metric_key: MetricKey,
    value: dict[MetricName, float],
    service_metrics: dict[MetricKey, dict[MetricName, float]],
):
    """Check if the metrics for a given service/domain are already in `service_metrics`. This can
    happen because the metrics file includes both service and domain metrics, which coincide for
    domains with just one service."""

    def _is_domain_key(metric_key: MetricKey) -> bool:
        return "_" not in metric_key

    def _has_service_key_equivalent(
        metric_key: MetricKey,
        service_metrics: dict[MetricKey, dict[MetricName, float]],
        value: dict[MetricKey, float],
    ) -> bool:
        for metric in service_metrics:
            if metric.startswith(metric_key):
                existing_jga = service_metrics[metric]["joint_goal_accuracy"]
                new_jga = value["joint_goal_accuracy"]
                return math.isclose(existing_jga, new_jga)
        return False

    if metric_key in service_metrics:
        return True
    if _is_domain_key(metric_key):
        return _has_service_key_equivalent(metric_key, service_metrics, value)
    return _has_service_key_equivalent(metric_key.split("_")[0], service_metrics, value)


def get_service_metrics(
    metrics: dict[MetricKey, dict[MetricName, float]]
) -> dict[MetricKey, dict[MetricName, float]]:
    """Extract service/domain level metrics from the metrics file."""

    service_metrics = {}
    for key, value in metrics.items():
        if key in ("#ALL_SERVICES", "#SEEN_SERVICES", "#UNSEEN_SERVICES"):
            continue
        if not is_present(key, value, service_metrics):
            service_metrics[key] = value

    return get_displayed_metrics(service_metrics)


def get_displayed_metrics(
    metrics: dict[MetricKey, dict[MetricName, float]]
) -> dict[MetricKey, dict[MetricName, float]]:
    """Filter the metrics for each key to extract only the relevant metrics."""
    filtered_metrics = {}
    for key, value in metrics.items():
        displayed_metrics = {}
        for m in DISPLAYED_METRICS:
            if m in value:
                displayed_metrics[m] = value[m]
            else:
                compatible_keys = [k for k in value if k.startswith(m)]
                for c_key in compatible_keys:
                    displayed_metrics[c_key] = value[c_key]
        filtered_metrics[key] = displayed_metrics
    return filtered_metrics


def format_percentage(value: float):
    """Convert value to percentage and truncate to 3 decimal places."""
    return f"{round(value * 100, 3)}%"


def display_aggregated_metrics(
    metrics: dict[AggregatedMetricKey, dict[MetricName, float]]
):
    """Display metrics in table format."""
    console = Console()
    table = Table(title="Aggregated Metrics")

    # Add columns
    table.add_column("Metric")
    for key in metrics.keys():
        table.add_column(key)

    # Add rows
    for metric_name in DISPLAYED_METRICS:
        is_joint_goal = metric_name == "joint_goal_accuracy"
        row_style = "bold underline" if is_joint_goal else None
        values = [
            format_percentage(metrics[key].get(metric_name, 0.0))
            for key in metrics.keys()
        ]

        table.add_row(metric_name, *values, style=row_style)

    console.print(table)


def display_summary_table(
    service_metrics: dict[str, dict[str, float]], metric_to_display: str
):
    """Display the summary table of the specified metric for each service."""
    table = Table(title=f"{metric_to_display.capitalize()} Summary")
    table.add_column("Service")
    table.add_column(metric_to_display, justify="right")

    for service_name, metrics in service_metrics.items():
        if metric_to_display not in metrics:
            compatible_metrics = [k for k in metrics if k.startswith(metric_to_display)]
            for m in compatible_metrics:
                table.add_row(f"{service_name}/{m}", format_percentage(metrics[m]))
        else:
            table.add_row(service_name, format_percentage(metrics[metric_to_display]))

    console = Console()
    console.print(table)


def display_detailed_metrics(service_name: str, metrics: dict[str, float]):
    """Display detailed metrics for a specific service."""
    table = Table(title=service_name)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for metric_name, metric_value in metrics.items():
        table.add_row(metric_name, format_percentage(metric_value))

    console = Console()
    console.print(table)


def display_service_metrics(service_metrics: dict[MetricKey, dict[MetricName, float]]):
    """Display the metrics for each service."""
    sorted_services = sorted(
        service_metrics.items(),
        key=lambda x: x[1]["joint_goal_accuracy"],
    )

    display_detail = (
        Prompt.ask(
            "Would you like to display detailed metrics for each service? (y/n): "
        ).lower()
        == "y"
    )

    if not display_detail:
        valid_metric = False
        while not valid_metric:
            metric_to_display = Prompt.ask("Enter the metric you want to display: ")
            first_service_metrics = service_metrics[sorted_services[0][0]]
            if metric_to_display in first_service_metrics or any(
                k.startswith(metric_to_display) for k in first_service_metrics
            ):
                valid_metric = True
            else:
                print("Invalid metric! Please enter a valid metric name.")
                retry = Prompt.ask("Do you want to retry? (y/n): ").lower()
                if retry != "y":
                    print("Exiting...")
                    return
        # Display the summary table for the selected metric
        sorted_service_metrics = {r[0]: service_metrics[r[0]] for r in sorted_services}
        display_summary_table(sorted_service_metrics, metric_to_display)
    else:
        for service_name, metrics in sorted_services:
            display_detailed_metrics(service_name, metrics)


def get_config_path() -> str:
    return str(resources.files("pytod.configs.apps") / ".")


@hydra.main(
    config_name="display_metrics.yaml",
    config_path=get_config_path(),
)
def display_metrics(cfg: DictConfig):
    metrics = load_json(cfg.file)
    aggregated_metrics = get_aggregated_metrics(metrics)
    display_aggregated_metrics(aggregated_metrics)
    service_metrics = get_service_metrics(metrics)
    display_service_metrics(service_metrics)


if __name__ == "__main__":
    display_metrics()
