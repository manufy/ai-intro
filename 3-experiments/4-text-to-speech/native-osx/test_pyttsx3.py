import pyttsx3

# Inicializar el motor de TTS
engine = pyttsx3.init()

# Obtener las propiedades disponibles del motor
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
voices = engine.getProperty('voices')

print(f"Rate: {rate}")
print(f"Volume: {volume}")
print(f"Voices: {[voice.id for voice in voices]}")

# Configurar una voz específica si es necesario
if voices:
    engine.setProperty('voice', voices[0].id)

# Texto a convertir en voz
text = "Hola, ¿cómo estás?"

# Usar el motor para decir el texto
engine.say(text)

# Esperar hasta que se haya dicho todo el texto
engine.runAndWait()
