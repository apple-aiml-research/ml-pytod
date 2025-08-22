#! /bin/bash
SGD_REPO='https://github.com/google-research-datasets/dstc8-schema-guided-dialogue'
SGD_STORE='data/raw/sgd'


# download SGD

if [ -z "$(ls -A $SGD_STORE)" ]; then
  echo "Cloning the Schema-Guided Dialogue (SGD) dataset"
  git clone $SGD_REPO data/raw/sgd
  rm -rf data/raw/sgd/.git
else
   echo "Directory $SGD_STORE not empty, skipping SGD download"
fi
