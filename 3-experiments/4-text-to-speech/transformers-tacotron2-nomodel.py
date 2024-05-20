from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

cache_dir = ".cache"

# Nombre del modelo TTS
model_name = "espnet/kan-bayashi_ljspeech_tts_train_tacotron2"

# Cargar el modelo y el tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir   )
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Crear un pipeline de text-to-speech
text_to_speech = pipeline("text2speech", model=model, tokenizer=tokenizer)

# Texto de ejemplo
text = "Hello, how are you?"

# Generar el audio
outputs = text_to_speech(text)

# Guardar el audio en un archivo
with open("output.wav", "wb") as f:
    f.write(outputs["audio"])
print("Audio guardado en output.wav")
