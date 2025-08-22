#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from copy import deepcopy
from typing import Any, Callable

from jinja2 import Environment, meta

from pytod.command import CommandCollection


class TemplateMixin:
    def __init__(
        self,
        schema: CommandCollection,
        template_factory: Callable[[], str] = lambda: "",
    ):
        self._template_generator = template_factory
        self._template_variables: set[str] = set()
        self._template = None
        self._schema = schema

    def _set_template_and_variables(self, environment: Environment):
        template_variables = meta.find_undeclared_variables(
            environment.parse(self._template_generator())
        )
        self._template = environment.from_string(self._template_generator())
        self._template_variables = template_variables.difference(EXCLUDE_VARIABLES)

    def _get_variables_from_kwargs(self, variable: str, request: Any, **kwargs) -> str:
        value = kwargs.get(variable)
        if value is None and getattr(self, variable, None) is None:
            raise AttributeError(
                f"Undefined variable {variable} for template {self.__class__.__name__}"
            )
        return str(value)

    def _get_template_variables(self, request: Any, **kwargs: Any) -> dict[str, Any]:
        """Resolve the variables from the prompt template.
        To be resolved, a template variable must
        have a corresponding method named `_get_${variable_name}`
        which returns its value.
        """
        template_variables: set = deepcopy(self._template_variables)
        vars, vals = [], []
        while template_variables:
            variable = template_variables.pop()
            try:
                value = getattr(self, f"_get_{variable}")(request, **kwargs)
            except AttributeError:
                try:
                    value = self._get_variables_from_kwargs(variable, request, **kwargs)
                except AttributeError:
                    if isinstance(request, dict):
                        value = request.get(variable, None)
                    else:
                        value = getattr(request, variable, None)
                    if value is None:
                        raise AttributeError(
                            f"Undefined variable {variable} for template {self.__class__.__name__}"
                        )
            vars.append(variable)
            vals.append(value)
        return dict(zip(vars, vals))

    @staticmethod
    def _get_triple_quote() -> str:
        return '"""'


EXCLUDE_VARIABLES = {
    "loop",
    "self",
    "context",
    "macros",
    "request",
    "session",
    "g",
    "url_for",
    "config",
}
