print("---- Cargando archivo de audio ----")
import librosa

# Carga el archivo de audio como mono
audio, _ = librosa.load("recording.wav", sr=16000, mono=True)

import matplotlib.pyplot as plt
import librosa.display

plt.figure().set_figwidth(12)
librosa.display.waveshow(audio, sr=16000)
plt.show()