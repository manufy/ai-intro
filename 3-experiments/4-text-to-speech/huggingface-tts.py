from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import pipeline

# Load the pre-trained TTS model and tokenizer
model_name = "facebook/wav2vec2-base-100h"
cache_dir = ".cache"
tokenizer = Wav2Vec2Processor.from_pretrained(model_name,  cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_name,  cache_dir=cache_dir)

# Create a text-to-speech pipeline
text_to_speech = pipeline("text-to-speech", model=model, tokenizer=tokenizer)

# Example usage
text = "Hello, how are you?"
text_to_speech(text, clean_up_tokenization_spaces=True, padding="longest", sampling_rate=16000)
