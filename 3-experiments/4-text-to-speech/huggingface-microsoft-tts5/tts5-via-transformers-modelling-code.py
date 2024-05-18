from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

cache_dir = ".cache"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts",  cache_dir=cache_dir)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts",  cache_dir=cache_dir)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",  cache_dir=cache_dir)

inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
