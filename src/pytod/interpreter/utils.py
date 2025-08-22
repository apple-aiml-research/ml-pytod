#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from typing import NamedTuple, Optional

from pytod.pytod_types.aliases import IntentName, ServiceName
from pytod.pytod_types.sgd_conversation import Author


class PolicyError(Exception):
    pass


class Tag(NamedTuple):
    source_author: Author
    tag: str
    service: Optional[ServiceName] = None
    intent: Optional[IntentName] = None
    annotated: bool = True

    def __eq__(self, other: "Tag") -> bool:
        return all((self.source_author == other.source_author, self.tag == other.tag))


class GrammarError(Exception):
    pass
