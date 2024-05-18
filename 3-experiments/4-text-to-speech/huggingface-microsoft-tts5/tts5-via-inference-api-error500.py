import requests
import soundfile as sf
import os

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 500 internal server error

API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	print(response.text)
 
	return response.content

audio_bytes = query({
	"inputs": "The answer to the universe is 42",
})
# You can access the audio with IPython.display for example
#from IPython.display import Audio
#Audio(audio_bytes)

print(audio_bytes)

# Guardar los bytes de audio en un archivo .wav
with open('output.wav', 'wb') as f:
    f.write(audio_bytes)