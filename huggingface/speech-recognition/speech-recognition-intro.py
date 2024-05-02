import torch
from transformers import pipeline
from datasets import load_dataset, Audio

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")


dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])