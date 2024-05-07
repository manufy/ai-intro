#/bin/bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && git checkout 2b26469
python3 llama.cpp/convert.py ./models/gpt4all-lora-quantized-ggml.bin