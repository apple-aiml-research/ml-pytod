#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal, NamedTuple, Optional, Union

from pytod.pytod_types.aliases import IntentName, ServiceName


class UserQuery(NamedTuple):
    query: str


@dataclass
class Response:
    response: str


NaturalLanguageTypes = Union[UserQuery, Response]


@dataclass(kw_only=True)
class PyTODInstructionMixin:
    # a semantic label of a PyTOD instruction
    # these are derived from the SGD dialogue
    # act ontology (see interpreter_ontology*.py)
    # Note this is a many-to-one mapping (different
    # dialogue acts map to the same statement)
    tag: str
    # whether the instruction corresponding
    # to this type can be eventually assigned
    # a variable in the transcript. A more
    # correct name would be "can_be_referenced"
    is_assignable: bool = True
    # whether variables corresponding to this
    # type can appear as positional arguments
    # to the NLG call
    pass_to_nlg: bool = False
    # whether the type appears in-line with
    # other statements or is just part of a
    # longer expression (NB: this *does not*
    # control assigment syntax)
    is_inlined: bool = False
    # filled to allow correct variable assignments
    # across domains and tasks
    service: Optional[ServiceName] = None
    intent: Optional[IntentName] = None


# noinspection PyDataclass
@dataclass(kw_only=True)
class ProgramStatement(PyTODInstructionMixin):
    expression: str
    pass_expression: bool = False
    variable: Optional[str] = None
    var_index: int = None
    metadata: Optional[dict] = None
    # information passed to the caller (eg
    # labels of different discourse phenomena)
    info: Optional[dict[str, Any]] = None


@dataclass(kw_only=True)
class BackendHint(PyTODInstructionMixin):
    dialog: str
    pass_to_nlg: bool = True
    metadata: Optional[dict] = None
    # the index of the variable
    # assigned to this hint
    var_index: int = None
    # a flag that tells the caller
    # that this object does not have an
    # expression attribute. It is used
    # to access `var_index` instead for the
    # purposes of collecting the arguments
    # for the NLG call
    pass_expression: bool = False
    origin: Optional[int] = None


@dataclass(kw_only=True)
class BackendNotification(PyTODInstructionMixin):
    dialog: str
    pass_to_nlg: bool = True
    metadata: Optional[dict] = None
    pass_expression: bool = False
    var_index: int = None
    origin: Optional[int] = None


PyTODInstruction = Union[
    ProgramStatement,
    BackendNotification,
    BackendHint,
]


@dataclass
class TemplateField:
    # indicates whether the field value
    # is a variable that should be looked
    # up inside Assignments while
    # rendering the template
    is_variable: bool
    field: str
    metadata: Optional[Any] = None
    resolve_with_metadata: bool = False
    requires_disambiguation: bool = False


# noinspection PyDataclass
@dataclass(kw_only=True)
class ProgramStatementTemplate(PyTODInstructionMixin):
    expression_template: str
    fields: list[TemplateField]
    requires_disambiguation: bool = False
    # information passed to the caller (eg
    # labels of different discourse phenomena)
    info: Optional[dict[str, Any]] = None
    # custom information used to support grammar
    # customisation
    metadata: Optional[dict[str, Any]] = None


@dataclass(kw_only=True)
class AttributeAccessTemplate(PyTODInstructionMixin):
    expression_template: str
    variable: str
    attribute: str
    fields: list[TemplateField]
    is_assignable: bool = False
    var_index: int = -1
    is_inlined: bool = True
    # indicates that the variable to be assigned depends
    # on the context (eg can refer to the current entity or
    # call or sth else). See `maybe_disambiguate` method of
    # Assignments() class
    requires_disambiguation: bool = False
    # which fields of the AttributeAccessTemplate need to be
    # updated with resolved fields
    disambiguate: Optional[
        list[Literal["variable"] | Literal["expression_template"]]
    ] = None


@dataclass(kw_only=True)
class BackendNotificationTemplate(PyTODInstructionMixin):
    dialog_template: str
    fields: list[TemplateField]
    pass_to_nlg: bool = True
    requires_disambiguation: bool = False
    origin_template: Optional[str] = None
    origin_fields: Optional[list[TemplateField]] = None


# noinspection PyDataclass
@dataclass(kw_only=True)
class BackendHintTemplate(PyTODInstructionMixin):
    dialog_template: str
    fields: list[TemplateField]
    pass_to_nlg: bool = True
    requires_disambiguation: bool = False
    origin_template: Optional[str] = None
    origin_fields: Optional[list[TemplateField]] = None

    def get_fields(self, fields: list[str]) -> Optional[list[TemplateField]]:
        to_return = list(
            chain(*[[f for f in self.fields if f.field == name] for name in fields])
        )
        return to_return or None


PyTODInstructionTemplate = Union[
    BackendHintTemplate,
    BackendNotificationTemplate,
    ProgramStatementTemplate,
    AttributeAccessTemplate,
]
