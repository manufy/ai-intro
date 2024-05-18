print ("----- setting up libraries -----")
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '.cache'

print ("----- setting up pipeline -----")


synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

print ("----- setting up dataset -----")


embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

print ("----- synthetizing audio -----")

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
