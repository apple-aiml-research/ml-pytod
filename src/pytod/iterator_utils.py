#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from collections import Counter
from typing import Optional

TASK_SERVICE_SEPARATOR = "|||"


def matches_service(task_sequence: str, service: Optional[str], **kwargs) -> bool:
    """Check if a conversation task sequence includes a service."""
    if service is None:
        return True
    if TASK_SERVICE_SEPARATOR in task_sequence:
        assert not task_sequence.endswith(TASK_SERVICE_SEPARATOR)
        task_sequence = task_sequence.split(TASK_SERVICE_SEPARATOR)
    else:
        task_sequence = [task_sequence]
    return any(f.startswith(service) for f in task_sequence)


def has_non_contiguous_service_access(
    task_sequence: str, service: Optional[str], **kwargs
) -> bool:
    """Checks if the user switches services between calls to search/transaction intents
    in the same service."""

    def extract_services(flow: list[str]) -> list[str]:
        services = [f.split("(")[0] for f in flow]
        assert not any(prev == next_ for prev, next_ in zip(services, services[1:]))
        return services

    if TASK_SERVICE_SEPARATOR not in task_sequence:
        return False

    services = extract_services(task_sequence.split(TASK_SERVICE_SEPARATOR))
    if len(services) == len(set(services)):
        return False

    service_counts = Counter(services)
    if service is not None:
        return service_counts[service] > 1
    return True


def contains_task(task_sequence: str, service: str, **kwargs) -> bool:
    """Check if a conversation flow includes a task.
    Task is represented as {service_name}({intent_name}
    where the placeholders are replaced with SGD schema
    element names."""
    return kwargs.get("contains_task", "") in task_sequence


def startswith_task(task_sequence: str, service: str, **kwargs) -> bool:
    """Check if a conversation flow includes a task.
    Task is represented as {service_name}({intent_name}
    where the placeholders are replaced with SGD schema
    element names."""
    return task_sequence.startswith(kwargs.get("startswith_task", "NOTSET"))
