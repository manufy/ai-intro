import requests
from pathlib import Path
from tqdm import tqdm

#  download the weights and use a script from the LLaMAcpp repository to convert the weights from the old format to the new one.
# It is a required step; otherwise, the LangChain library will not identify the checkpoint file.
# https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/
# wget https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin

local_path = './.cache/models/gpt4all-lora-quantized-ggml.bin'
Path(local_path).parent.mkdir(parents=True, exist_ok=True)

url = 'https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin'

# send a GET request to the URL to download the file.
response = requests.get(url, stream=True)

# open the file in binary mode and write the contents of the response
# to it in chunks.
with open(local_path, 'wb') as f:
    for chunk in tqdm(response.iter_content(chunk_size=8192)):
        if chunk:
            f.write(chunk)