#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from textwrap import dedent


def shown_descriptions_display():
    template = """\
    {%- if item.displayed_documentation -%}
        {%- for arg in item.displayed_documentation %}
        {{ arg }}: {{arg | get_arg_type(item.task_schema) }}
            {{ arg | get_arg_description(item.task_schema) }}
        {%- endfor %}
    {%- endif %}
    """
    return dedent(template)


def hidden_descriptions_display():
    template = """\
    {%- if item.hidden_documentation %}
        {{ item.hidden_documentation | maybe_add_arg_type(item.task_schema) | join(', ') }}
            see task #{{item.task_referenced }} documentation
    {%- endif %}
    """
    return dedent(template)


def selected_entity_template_factory() -> str:
    """Render an entity selected by the user. This is displayed if
    symbolic values (ie object references in value) are configured.

    Examples
    --------
    1. If some properties are shared with another completed task, a task
    reference is displayed instead of the property descriptions.

    class therapist:
    \"\"\"Object returned by the `services_4_find_provider` API. The following properties may
    be referenced in subsequent API calls.

    Properties
    ----------
    therapist_name: str
        name of the therapist
    address, city
        see task #1 documentation
    \"\"\"

      class therapist:
    \"\"\"Object returned by the `services_4_find_provider` API. The following properties may
    be referenced in subsequent API calls.

    Properties
    ----------
    therapist_name: str
        name of the therapist
    address, city
        see task #1 documentation
    \"\"\"

    2. Otherwise, all properties are displayed

    class therapist:
    \"\"\"Object returned by the `services_4_find_provider` API. The following properties may
    be referenced in subsequent API calls.

    Properties
    ----------
    therapist_name: str
        name of the therapist
    city: str
        area where user wants to search for a therapist
    address: str
        address of the therapist
    \"\"\"

    3. If there are no relevant properties other APIs can reference, nothing is displayed

    Checked alarms set on their device. The alarm_1_get_alarms API returned a `alarm`
    object (x1).

    class alarm:
        \"\"\"Object returned by the `alarm_1_get_alarms` API.\"\"\"


    """
    template = """\
    {{ task_completion_summary }} The {{ tool_name }} API returned a '{{ entity_name }}' object ({{ entity_variables | join(', ') }}).

    class {{ entity_name }}:
        \"""Object returned by the '{{ tool_name }}' API. {%- if item.requires_property_display %} The following properties may be referenced instead of copying keyword values in subsequent API calls. {%- else %}{{ triple_quote }}{%- endif %}

        {%- if item.requires_property_display %}

        Properties
        ----------{{ item | docs_formatter | join}}
        \"""
        {%- endif %}
    """  # noqa
    return dedent(template)


def selected_entity_template_factory_resolved_carryover_arg_values() -> str:
    """Like selected_entity_template_factory, but displayed if the
    carried over values are represented in resolved form (ie as strings
    and not object references)."""
    template = """\
    {{ task_completion_summary }} The {{ tool_name }} API returned a '{{ entity_name }}' object ({{ entity_variables | join(', ') }}).

    class {{ entity_name }}:
        \"""Object returned by the '{{ tool_name }}' API. {%- if item.requires_property_display %} In subsequent API calls, you should pass, to compatible arguments, the last value mentioned for the following entity properties: {%- else %}{{ triple_quote }}{%- endif %}

        {%- if item.requires_property_display %}

        Properties
        ----------{{ item | docs_formatter | join}}
        \"""
        {%- endif %}
    """
    return dedent(template)


def complete_transaction_template_factory() -> str:
    """Template rendered when a transaction is complete, if symbolic values (ie
    values contain object references) are configured."""
    template = """\
    {{ task_completion_summary}} {%- if item.requires_property_display %} The following '{{ tool_name }}' ({{ entity_variables | join(', ') }}) properties may be referenced in subsequent API calls instead of copying keyword values because the user has mentioned or confirmed during this or previous tasks: {%- endif %}

    {%- if item.requires_property_display %}

    class {{ tool_name }}:
        \"""{{item.task_schema.description | lower }}.

        Properties
        ----------{{ item | docs_formatter | join}}
        \"""
    {%- endif %}
    """  # noqa
    return dedent(template)


def complete_transaction_template_factory_resolved_carryover_arg_values() -> str:
    """Template rendered as soon as a transaction is complete, when the
    language model resolves carried-over values to natural language."""
    template = """\
    {{ task_completion_summary}} {%- if item.requires_property_display %} The '{{ tool_name }}' ({{ entity_variables | join(', ') }}) properties below have already been mentioned or confirmed by the user in this or previous tasks. They should be copied as values of compatible arguments in subsequent API calls. {%- endif %}

    {%- if item.requires_property_display %}

    class {{ tool_name }}:
        \"""{{item.task_schema.description | lower }}.

        Properties
        ----------{{ item | docs_formatter | join}}
        \"""
    {%- endif %}
    """  # noqa
    return dedent(template)


def task_stack_template_factory() -> str:
    template = """\
    {%- if task_completion_stack | length > 0 -%}
    The user has completed the following tasks:
    {%- for item in task_completion_stack %}
    #{{ loop.index }}. {{ item | completed_task_formatter }}
    {%- endfor %}
    {%- endif %}

    Your task is to identify which of the following tasks the user wants to complete, and update it accordingly as the user provides more information during the conversation. You should also carefully follow any developer instructions to answer user questions and complete follow-up tasks.
    """
    return dedent(template)


def developer_turn_following_selection() -> str:
    template = """\
    {%- if object_references | length > 0 -%}
    developer: In future API calls, you can reference {{ object_references[0].reference | extract_variable }} followed by one of the properties below to express keywords values the user or agent have mentioned while completing {{ object_references[0].active_intent }} task.
    {%- for ref in object_references %}
    - {{ ref | object_ref_formatter}}
    {%- endfor %}
    {%- else %}:
    {%- endif %}
    """
    return dedent(template)


def developer_turn_following_iteration() -> str:
    template = """\
    {%- if requestable_slots | length > 0 -%}
    developer: to answer questions about {{ requestable_slots | inline_requestables }} pass the relevant property or properties to '{{ nlg_call_tool }}' (eg, {{ usage_example }}).
    {%- endif %}
    """  # noqa
    return dedent(template)


def developer_turn_following_iteration_list() -> str:
    template = """\
    {%- if requestable_slots | length > 0 -%}
    developer: the user may request specific information about {{ entity_info }} properties listed below. Pass the relevant property or properties to '{{ nlg_call_tool }}' to answer (eg, {{ usage_example }}).
    {%- for prop in requestable_slots %}
    - {{ prop | prop_formatter }}
    {%- endfor %}
    {%- endif %}
    """  # noqa
    return dedent(template)


def developer_turn_following_confirmation() -> str:
    template = """\
    {%- if requestable_slots | length > 0 -%}
    developer: unless a signal indicates a '{{ transaction }}' calling error with no alternative arrangements, the properties
    {%- for prop in requestable_slots %}
    - {{ prop | prop_formatter }}
    {%- endfor %}
    may be communicated to the user upon their request by referencing '{{ object_ref }}' while calling '{{ nlg_call_tool }}' (eg, {{ usage_example }}).
    {%- endif %}
    """  # noqa
    return dedent(template)


def developer_turn_confirmed_argument_instructions() -> str:
    template = """\
    developer: in the event of a '{{ transaction }}' failure, pass the following keywords (confirmed by the user) to subsequent '{{ transaction }}' calls:
    {%- for ref in object_references %}
    - {{ ref.reference }}
    {%- endfor %}
    """  # noqa
    return dedent(template)


def developer_turn_confirmed_argument_instructions_resolved_carryover_arg_values() -> (
    str
):
    template = """\
    developer: in the event of a '{{ transaction }}' failure, copy the values of the keywords below the agent last mentioned to subsequent '{{ transaction }}' calls:
    {%- for ref in object_references %}
    - {{ ref.arg_name[1] }}
    {%- endfor %}
    """  # noqa
    return dedent(template)


def nlu_example_template():
    template = """\
    Q: Answer the following questions. Output "unanswerable" if the question cannot be answered given the conversation.

    In the conversation:

    {{source_turns | conversation_formatter }}

    {%- for item in slot_list %}
    {{ loop.index }}) {{ item | question_formatter }}
    {%- endfor %}

    Answer:
    """
    return dedent(template)
