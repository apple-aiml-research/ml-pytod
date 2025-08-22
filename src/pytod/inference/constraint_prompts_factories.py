#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from textwrap import dedent


def hallucinated_arg_name_template_factory() -> str:
    """A simple prompt for mapping hallucinated, but semantically
    meaningful slots to API arguments.

    - {{ predicted_argument }} is replaced with the name of a hallucinated slot
    - the {{ schema }} field is replaced with slot: definition and a special
    clause that allows the model to ignore the argument.

    Example
    -------

    Given the definitions, which keyword below best matches 'from_station'?

       - a) journey_starts_from: starting city for train journey
       - b) to: ending city for train journey
       - c) journey_start_time: time of start of train journey
       - d) number_of_adults: int
       - e) trip_protection: true|false
       - f) ticket_fare_class: value|business
       - g) none: the definitions do not describe 'from_station'

    Answer:

    Notes
    -----
    1. A definition is added only when constraining value object
    references. See `SlotConstraintRequest` definition.

    """
    template = """\
    Given the definitions, which keyword below best matches {{ predicted_argument_with_definition }}?

        {% for slot in slot_schemas -%}
        - {{ loop.index | int2alpha }}) {{ slot | slot_definition_formatter }}
        {%if loop.last -%}
        - {{ (loop.index + 1) | int2alpha }}) none: the definitions do not describe {{ predicted_argument }}
        {% endif -%}
        {%- endfor %}
    Answer:"""  # noqa
    return dedent(template)


def memorised_arg_name_template_factory() -> str:
    """A prompt for mapping memorised argument names from the
    same domain to schema arguments. The template contains:

    -  the description of the memorised slot in {description} ({slot_name}) format.
    -  the description of slots in the schema the slot may be mapped to.
    -  the name of the slot memorised

    Example
    -------
    Which sentence below is a paraphrase of 'date of car rental pickup' (pickup_date)?

        - a) city where you want to rent the car
        - b) the first date to start using the rental car
        - c) the date to return the car
        - d) type of the car (hatchback, sedan, suv)
        - e) place to pick up the car
        - f) whether to purchase insurance
        - g) no option above paraphrases 'pickup_date'

    Answer:
    """
    template = """\
    Which sentence below paraphrases {{ memorised_argument_description_with_name }}?

        {% for slot in slot_schemas -%}
        - {{ loop.index | int2alpha }}) {{ slot | slot_definition_formatter }}
        {%if loop.last -%}
        - {{ (loop.index + 1) | int2alpha }}) none of the sentences above paraphrases {{ memorised_argument_name }}
        {% endif -%}
        {%- endfor %}
    Answer:
    """  # noqa
    return dedent(template)


def hallucinated_arg_name_with_cat_value_template_factory() -> str:
    """A prompt for mapping hallucinated slots for which a categorical
    value (or part thereof) was predicted to a categorical slot-value
    pair.

    Notes
    -----
    1. Categorical also includes boolean slots in this case.

    Example
    ------
    Here are some definitions:

        - subtitle_language: language to use for subtitles (or none for no subtitles)

    Given these, 'subtitle_free = true' is a synonym of:

        - a) subtitle_language = none
        - b) subtitle_language = english
        - c) subtitle_language = mandarin
        - d) subtitle_language = spanish
        - e) options do not describe 'subtitle_free = true'

    Answer:
    """
    template = """\
    Here are some definitions:
    {% for slot in slot_schemas %}
        - {{ slot | slot_definition_formatter }}
    {%- endfor %}
    {% set cnt = [0] %}
    Given these, {{ predicted_keyword }} is a synonym of:
        {% for slot in slot_schemas -%}
        {% for value in slot.possible_values -%}
        {% if cnt.append(cnt.pop()  + 1) %}{% endif %}
        - {{ cnt[0] | int2alpha }}) {{ slot.name }} {{ "=" }} {{ value | lower }}
        {%- endfor -%}
        {%if loop.last %}
        - {{ (cnt[0] + 1) | int | int2alpha }}) options do not describe {{ predicted_keyword }}
        {% endif -%}
        {%- endfor %}
    Answer:
    """
    return dedent(template)


def bool_enum_arg_value_template_factory() -> str:
    """A prompt for mapping hallucinated bool and enum argument
    values for which the argument was correctly predicted to a possible
    value.

    Example
    ------
    Here are some definitions:

        - payment_method: the source of money used for making the payment
    Given these, 'payment_method = master_card' is a synonym of:

        - a) payment_method = app balance
        - b) payment_method = debit card
        - c) payment_method = credit card

    Answer:
    """
    template = """\
    Here are some definitions:

        - {{ arg_def.name }}: {{ arg_def.description | lower }}

    Given these, {{ predicted_keyword }} is a synonym of:
        {% for value in arg_def.possible_values %}
        - {{ loop.index | int2alpha }}) {{ arg_def.name }} {{ "=" }} {{ value | lower }}
        {%- endfor %}

    Answer:
    """
    return dedent(template)
