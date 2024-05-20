import requests

API_URL = "https://api-inference.huggingface.co/models/espeak-ng"
headers = {"Content-Type": "application/json"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Esto lanzará una excepción si el código de estado no es 200
    return response.content

audio_bytes = query({
    "inputs": "The answer to the universe is 42",
})

# Guardar los bytes de audio en un archivo .wav
with open('output.wav', 'wb') as f:
    f.write(audio_bytes)
print("Audio guardado en output.wav")
