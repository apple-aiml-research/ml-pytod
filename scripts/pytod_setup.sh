#! /bin/bash

# correct annotations (both dialogue and schema)
python scripts/pytod_setup/correct_annotations.py --multirun split=train,dev,test

# build canonical value lookup for normalisation
python scripts/pytod_setup/build_normalisation_table.py
dotenv set NORMALISATION_LOOKUP "${PWD}"/resources/normalisation/sgd_canonical_value_map.json

# extend slot value annotations of system-side slots with normalised values
python scripts/pytod_setup/extend_state_annotation.py --multirun split=train,dev,test

# augment schema with metadata necessary for building commands
python scripts/pytod_setup/augment_schema.py

# extract entities from dialogues for db building purposes
python scripts/pytod_setup/gather_entities_from_annotations.py --multirun split=dev,test
dotenv set ENTITIES "${PWD}"/resources/entities

# gather api responses from dialogues for simulating transactions
python scripts/pytod_setup/gather_transaction_responses.py --multirun split=dev,test
dotenv set API_RESPONSES "${PWD}"/resources/transaction_responses

# create templates for storing model predictions
python scripts/pytod_setup/create_templates.py --multirun split=dev,test

# annotate frames with metadata for computing system task completion
python scripts/pytod_setup/annotate_task_completion.py --multirun split=dev,test

# sample a diverse range of dialogues (used eg, to manually validate data processing)
python scripts/pytod_setup/sample_diverse_dialogues.py
