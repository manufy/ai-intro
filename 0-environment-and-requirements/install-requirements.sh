#!/bin/bash
./create-conda-env.sh
./activate-conda-env.sh
pip install -r activeloop-requirements.txt
pip install -r huggingface-llm-requirements.txt
pip install -r huggingface-speeech-recognition-requirements.txt