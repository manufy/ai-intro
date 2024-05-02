#!/bin/bash
./create-conda-env.sh
./activate-conda-env.sh
pip install -r activeloop-requirements.txt
pip install -r huggingface-requirements.txt