#  pip install pyobjc pyttsx3
# pip install pyobjc-core pyobjc-framework-Cocoa

import pyttsx3

# Inicializar el motor de TTS
engine = pyttsx3.init()

# Texto a convertir en voz
text = "Hola, ¿cómo estás?"

# Usar el motor para decir el texto
engine.say(text)

# Esperar hasta que se haya dicho todo el texto
engine.runAndWait()