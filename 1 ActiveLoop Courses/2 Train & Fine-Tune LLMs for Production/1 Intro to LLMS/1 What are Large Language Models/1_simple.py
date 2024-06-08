
import os
import openai

# English text to translate
english_text = "Hello, how are you?"

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f'Translate the following English text to French: "{english_text}"'}
  ],
)

print(response['choices'][0]['message']['content'])