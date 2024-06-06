import whisper

model = whisper.load_model("base")
result = model.transcribe("lecuninterview.mp4")
print(result['text'])

with open ('text.txt', 'w') as file:  
    file.write(result['text'])