import requests

API_URL = "https://huggingface.co/spaces/coqui-ai/TTS/+/api/predict"
headers = {"Content-Type": "application/json"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Esto lanzará una excepción si el código de estado no es 200
    return response.json()

# Formato de la carga útil para el espacio de coqui-ai/TTS
payload = {
    "data": ["The answer to the universe is 42"]
}

result = query(payload)

# El audio está en result['data'] que contiene una lista de resultados.
# Aquí asumimos que el primer resultado es el audio deseado.
audio_url = result['data'][0]
audio_bytes = requests.get(audio_url).content

# Guardar los bytes de audio en un archivo .wav
with open('output.wav', 'wb') as f:
    f.write(audio_bytes)
print("Audio guardado en output.wav")
