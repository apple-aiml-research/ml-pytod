[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/pytod.svg?branch=main)](https://cirrus-ci.com/github/<USER>/pytod)
[![ReadTheDocs](https://readthedocs.org/projects/pytod/badge/?version=latest)](https://pytod.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/pytod/main.svg)](https://coveralls.io/r/<USER>/pytod)
[![PyPI-Server](https://img.shields.io/pypi/v/pytod.svg)](https://pypi.org/project/pytod/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/pytod.svg)](https://anaconda.org/conda-forge/pytod)
[![Monthly Downloads](https://pepy.tech/badge/pytod/month)](https://pepy.tech/project/pytod)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/pytod)
-->

# pytod

_pytod_ is a library which provides a simulated environment for the Schema-Guided Dialogue (SGD) dataset (_Rastogi et al., 2019_). It simulates SGD APIs, including database responses and API behavior, according to the complex policies inherent in the dataset, providing a resource for conversational tool-use and zero-shot end-to-end task-oriented dialogue research.

Accompanies the paper [<i>PyTOD: Programmable Task-Oriented Dialogue with Execution Feedback</i>](https://arxiv.org/abs/2508.15456).

## Installation

In order to set up the necessary environment:

1. While in the `pytod` root directory, run the following commands:

````bash
pipenv install
````

> **_NOTE:_**  The pipenv environment has `pytod` installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pipenv install.` again. Alternatively, simply use `pipenv
> update` to add new dependencies to the virtual environment.

To activate the project's virtualenv run

```bash
pipenv shell
```

and you can run commands inside the virtual environment with

```bash
pipenv run
```
Add

```bash
eval "$(_PIPENV_COMPLETE=zsh_source pipenv)"
```

to `~/.zshrc` to get shell completion.

2. For contributing, install the developer requirements via

```bash
pipenv install --dev tox sphinx pre-commit pytest
```

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

## Downloading the data

1. Download the Schema Guided Dialogue (SGD) dataset [[1]](https://arxiv.org/abs/1909.05855).

```bash
chmod +X scripts/fetch_data.sh && bash scripts/fetch_data.sh
```

## Setting up the pytod environment

This step should only be performed once upon installation. Run

```bash
chmod +X scripts/pytod_setup.sh && bash scripts/pytod_setup.sh
```

This will apply various corrections to the data, build databases and API simulations, normalisation tables and so on.


## Converting SGD dialogue files to PyTOD transcripts

You can convert the SGD training set dialogue files to PyTOD transcripts, with the following command


```bash
pipenv run python scripts/create_pytod_dialogues.py \
filters=v0.6.1 version=v0.9.1 interpreter=v0.5 \
interpreter/backend/actions=v0.3.4 \
interpreter.backend.show_values_samples_in_slot_filling_hints=false \
interpreter.backend.include_slot_value_references_in_confirmation_hints=false \
interpreter.backend.include_slot_value_references_in_alternative_hints=false \
interpreter.backend.multiple_confirmation_hints=true \
interpreter.user.reference_command_for_wildcard_carryover=true \
interpreter.user.value_carryover='natural_language' \
interpreter.user.user_task_retry_instruction_format=call \
interpreter.user.user_query_retry_instruction_format=call \
interpreter.copy_selected_entities_to_search=true \
interpreter.resolve_carry_over_to_entity=true \
split=train --config-name=v3
```

Set `split=dev` and `split=test` to process the development and test sets.


To print the transcripts specify one or multiple PyTOD transcripts in human-readable format use the `display_pytod` endpoint:

```bash
pipenv run display_pytod split=dev ids="['1_00000']" version=v0.9.1
```

The `ids` are the transcript IDs, which match the `dialogue_id` in the SGD `dialogue_*.json` files. This command will display a transcript as follows:


<pre style="background-color:#1e1e1e;color:#d4d4d4;padding:15px;border-radius:5px;overflow:auto;">
<span style="color:#f78c6c;">user</span>: I want to make a restaurant reservation for 2 people at half past 11 in the morning.
<span style="color:#82aaff;">x1</span> <span style="color:#c3e88d;">restaurants_2_reserve_restaurant</span>(<span style="color:#82aaff;">time</span> = <span style="color:#c792ea;">'half past 11 in the morning'</span>,
   <span style="color:#82aaff;">number_of_seats</span> = <span style="color:#f78c6c;">2</span>)
<span style="color:#82aaff;">x2</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'request value: restaurant_name'</span>)
<span style="color:#82aaff;">x3</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'request value: location'</span>)
<span style="color:#82aaff;">x4</span> <span style="color:#ffcb6b;">say</span>(x2, x3)
<span style="color:#c792ea;">agent</span>: What city do you want to dine in? Do you have a preferred restaurant?
<span style="color:#f78c6c;">user</span>: Please find restaurants in San Jose. Can you try Sino?
<span style="color:#82aaff;">x5</span> <span style="color:#82aaff;">x1</span>.<span style="color:#82aaff;">location</span> = <span style="color:#c792ea;">'San Jose'</span>; <span style="color:#82aaff;">x1</span>.<span style="color:#82aaff;">restaurant_name</span> = <span style="color:#c792ea;">'Sino'</span>
<span style="color:#82aaff;">x6</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'ask the user to confirm: date'</span>)
<span style="color:#82aaff;">x7</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'ask the user to confirm: number_of_seats'</span>)
<span style="color:#82aaff;">x8</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'ask the user to confirm: restaurant_name'</span>)
<span style="color:#82aaff;">x9</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'ask the user to confirm: time'</span>)
<span style="color:#82aaff;">x10</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'ask the user to confirm: location'</span>)
<span style="color:#82aaff;">x11</span> <span style="color:#ffcb6b;">say</span>(x6, x7, x8, x9, x10)
<span style="color:#c792ea;">agent</span>: Confirming: I will reserve a table for 2 people at Sino in San Jose. The reservation time is 11:30 am today.
<span style="color:#f78c6c;">user</span>: Yes, thanks. What's their phone number?
<span style="color:#82aaff;">x12</span> <span style="color:#c3e88d;">confirm</span>(x1)
<span style="color:#82aaff;">x13</span> <span style="color:#c3e88d;">perform</span>(x1)
<span style="color:#82aaff;">x14</span> <span style="color:#ffcb6b;">say</span>(x1.phone_number, x13)
<span style="color:#c792ea;">agent</span>: Your reservation has been made. Their phone number is 408-247-8880.
<span style="color:#f78c6c;">user</span>: What's their address? Do they have vegetarian options on their menu?
<span style="color:#82aaff;">x15</span> <span style="color:#ffcb6b;">say</span>(x1.has_vegetarian_options, x1.address)
<span style="color:#c792ea;">agent</span>: The street address is 377 Santana Row #1000. They have good vegetarian options.
<span style="color:#f78c6c;">user</span>: Thanks very much.
<span style="color:#82aaff;">x16</span> <span style="color:#ffcb6b;">conversation_pause</span>()
<span style="color:#82aaff;">x17</span> <span style="color:#ffcb6b;">Hint</span>(<span style="color:#c792ea;">'ask the user if they require further assistance'</span>)
<span style="color:#82aaff;">x18</span> <span style="color:#ffcb6b;">say</span>(x17)
<span style="color:#c792ea;">agent</span>: Is there anything else I can help you with?
<span style="color:#f78c6c;">user</span>: No, that's all. Thanks.
<span style="color:#82aaff;">x19</span> <span style="color:#ffcb6b;">say</span>()
<span style="color:#c792ea;">agent</span>: Have a great day.
</pre>


### Fine-tuning PyTOD

If you wish to finetune PyTOD, you may run the following commands to obtain:

1. Action Parser Examples

```bash
pipenv run python scripts/create_pytod_text2text_examples.py \
split=train version=v0.9.1 patch_version=1  \
text2text_conversion=rendered_entities_resolved_values \
history_processor=rendered_entities \
multidomain_prompts=true \
debug=true
```

and set `split=dev` and `split=test` to process the other splits.


2. Parser supervisor examples

```bash
pipenv run python scripts/create_nlu_text2text_examples.py \
  split=train formatter.randomise_prompt_elements=true \
  formatter.unk_slot_probability=0.5 \
  formatter.similar_slot_probability=0.5 \
  version=v0.9.1 patch_version=1
```

You should then finetune your model on both datasets.

## `pytod` simulation 

The simulation environment is implemented in the `simulation` package. A service is defined as a collection of
APIs which can be search APIs which provide interfaces to databases (e.g., `FindBus`) or APIs that allow users
to execute transactions (`BuyBusTicket`). An SGD service is implemented as follows:

<pre style="background-color:#2b2b2b;color:#a9b7c6;padding:15px;border-radius:5px;overflow:auto;">
<span style="color:#cc7832;">@register_command</span>(service=<span style="color:#6a8759;">"Events_1"</span>)
<span style="color:#ffc66d;">class</span> <span style="color:#a9b7c6;">FindEvents</span>(<span style="color:#9876aa;">SearchCommand</span>):
    <span style="color:#9876aa;">category</span>: <span style="color:#6897bb;">SearchCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">SearchCommandArgument</span>()
    <span style="color:#9876aa;">subcategory</span>: <span style="color:#6897bb;">SearchCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">SearchCommandArgument</span>()
    <span style="color:#9876aa;">city_of_event</span>: <span style="color:#6897bb;">SearchCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">SearchCommandArgument</span>()
    <span style="color:#9876aa;">date</span>: <span style="color:#6897bb;">SearchCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">SearchCommandArgument</span>()

    <span style="color:#ffc66d;">def</span> <span style="color:#a9b7c6;">__init__</span>(<span style="color:#9876aa;">self</span>, <span style="color:#9876aa;">dialogue_id</span>: <span style="color:#6897bb;">DialogueID</span>):
        <span style="color:#ffc66d;">super</span>().<span style="color:#a9b7c6;">__init__</span>(dialogue_id)


<span style="color:#cc7832;">@register_command</span>(service=<span style="color:#6a8759;">"Events_1"</span>)
<span style="color:#ffc66d;">class</span> <span style="color:#a9b7c6;">BuyEventTickets</span>(<span style="color:#9876aa;">ConfirmedCommand</span>):
    <span style="color:#9876aa;">event_name</span>: <span style="color:#6897bb;">ConfirmedCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">ConfirmedCommandArgument</span>()
    <span style="color:#9876aa;">number_of_seats</span>: <span style="color:#6897bb;">ConfirmedCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">ConfirmedCommandArgument</span>()
    <span style="color:#9876aa;">date</span>: <span style="color:#6897bb;">ConfirmedCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">ConfirmedCommandArgument</span>()
    <span style="color:#9876aa;">city_of_event</span>: <span style="color:#6897bb;">ConfirmedCommandArgument</span>[<span style="color:#ffc66d;">str</span>] = <span style="color:#6897bb;">ConfirmedCommandArgument</span>()

    <span style="color:#ffc66d;">def</span> <span style="color:#a9b7c6;">__init__</span>(<span style="color:#9876aa;">self</span>, <span style="color:#9876aa;">dialogue_id</span>: <span style="color:#6897bb;">DialogueID</span>):
        <span style="color:#ffc66d;">super</span>().<span style="color:#a9b7c6;">__init__</span>(dialogue_id)
</pre>

Here `SearchCommand` and `ConfirmedCommand` implement the SGD policy graph. In other words, when called, these objects return
the system actions (e.g, `Hints`) which guide agents according to the SGD policy. `SearchCommandArgument` and `ConfirmedCommandArgument` are descriptors, which
perform post-processing operations which convert LM outputs to schema compatible values (e.g., type coercion) and can be configured to provide feedback. 

The package contains:

- Simulations for the SGD transactional APIs (`api_driver.py`)
- A `mongoquery` database simulation (`database.py`) for all the SGD databases
- Code to parse entities from `json` dicts to `python` objects

## `pytod` execution

This package implements the `PyTOD` agent. The `execute_instructions` method takes as input a list of program
statements generated by the action parser, returning a list of system actions that guide the agent according to the SGD policy.

We provide the `OfflineSessionExecutor` helper which takes as an input a PyTOD transcript and returns the conversation state
the format required by the official DSTC8 evaluator. The commands

```bash
pipenv run python scripts/evaluate_pytod.py split=dev version=v0.9.1 
```

```bash
pipenv run python scripts/evaluate_pytod.py split=test version=v0.9.1 
```

use it to execute all the ground truth PyTOD transcripts for the development and test sets.§

## Project Organization


```
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── Pipfile                 <- For virtual environment management, contains abstract dependencies
├── Pipfile.lock            <- Precise description of the dependency tree, enables recreating the environment elsewhere
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── resources
│   └── lexical             <- String equivalence maps, including semantically aware ones
│      └── mining_results   <- outputs of data mining steps necessary to understand annotation patterns
│      └── sgd
│          └── index        <- Tree splitting conversations according to the conversation structure
├── scripts                 <- Analysis and production scripts which import the actual PYTHON_PKG,
│    └── data_mining        <- Scripts for analysing annotation patterns
│    └── scratches          <- quick & dirty ones
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── pytod          
│       └── apps            <- CLI entry points implementation
│       └── configs         <- configurations library scripts and apps
│           └── apps        <- endpoints configs 
│           └── pytod       <- PyTOD transcript generation configuration group
│           └── pytod_finetuning <- configs for converting data to text2text format for PyTOD finetuning
│           └── pytod_setup <- configs for scripts that prepare the simulation environment 
│       └── evaluation      <- DSTC8 evaluator code, adapted for PyTOD evaluation.
│       └── execution       <- Implementation of the execution engine 
│       └── interpreter     <- PyTOD interpreter, converts SGD annotation to PyTOD transcripts
│       └── parser          <- pydantic validators, used by the dialogue manager to ensure AP outputs are well-formed and syntactically correct
│       └── prompting       <- formatter classes for converting transcripts from structured to text-to-text (source-target) format
│       └── simulation      <- implements simulated environment for SGD, simulation of the policy graph
│         └── services      <- concrete service implementations  
│       └── toolbox         <- Alternative SGD schema representation, used by the PyTOD interpreter 
│       └── pytod_types     <- `pydantic` classes, defining the data model used by the `pytod` interpreter
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```
