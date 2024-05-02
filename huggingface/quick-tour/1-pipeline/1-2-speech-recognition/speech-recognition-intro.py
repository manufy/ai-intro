import torch
from transformers import pipeline
from datasets import load_dataset, Audio

# The pipeline() can also iterate over an entire dataset for any task you like. 
# For this example, let’s choose automatic speech recognition as our task:
    
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Load an audio dataset (see the 🤗 Datasets Quick Start for more details) you’d like to iterate over. 
# For example, load the MInDS-14 dataset

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

# You need to make sure the sampling rate of the dataset matches the sampling rate facebook/wav2vec2-base-960h was trained on:

dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

# The audio files are automatically loaded and resampled when calling the "audio" column. 
# Extract the raw waveform arrays from the first 4 samples and pass it as a list to the pipeline:

result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])

# For larger datasets where the inputs are big (like in speech or vision), you’ll want to pass a generator instead of a list to load all the inputs in memory.
# Take a look at the pipeline API reference for more information.