# Use a pipeline as a high-level helper
# pip install tf-keras sounddevice librosa

# ignore tensorflow warnings regarding CPU instrucyions 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


print("---- Importing transformers pipeline ----")

from transformers import  pipeline

print ("---- Importing AutoModel and AutoTokenizer ----")

from transformers import AutoModel,AutoTokenizer

print ("---- Setting up AutoModel ----")

model = AutoModel.from_pretrained("openai/whisper-large-v3", cache_dir="./cache")

print ("---- Setting up AutoTokenizer ----")

# Descarga el tokenizador
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3", cache_dir="./cache")

from transformers import Wav2Vec2FeatureExtractor

# Descarga el extractor de características

print ("---- Setting up Feature Extractor ----")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("openai/whisper-large-v3", cache_dir="./cache")

# Usa el modelo en la pipeline

print ("---- Using pipeline ----")

pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

# Load model directly
#from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

#processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", cache_dir="./cache")
#model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3", cache_dir="./cache")

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Graba audio desde el micrófono

print("---- Grabando ----")

duration = 5  # segundos
fs = 16000  # frecuencia de muestreo
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # espera hasta que termine la grabación

# Convierte la grabación a mono si no lo es ya
if recording.ndim > 1:
    recording = np.mean(recording, axis=1)

# Guarda la grabación en un archivo WAV
write("recording.wav", fs, recording)

# Lee el archivo WAV
#input_audio = feature_extractor.from_file("recording.wav", return_tensors="pt")
# aqui usar librosa para cargar el archivo de audio
# Carga el archivo de audio


print("---- Cargando archivo de audio ----")
import librosa

# Carga el archivo de audio como mono
audio, _ = librosa.load("recording.wav", sr=16000, mono=True)



# Usa el extractor de características para procesar el audio

print ("---- Procesando audio ----")



input_values = feature_extractor(audio, return_tensors="pt" , sampling_rate=16000).input_values


# Convierte el tensor de PyTorch a un array de numpy
input_values = input_values.numpy()


print(input_values)
# Usa la pipeline para reconocer el audio

import matplotlib.pyplot as plt
import librosa.display

plt.figure().set_figwidth(12)
librosa.display.waveshow(input_values, sr=16000)

print("---- Reconociendo ----")

transcription = pipe(input_values)

print(transcription)