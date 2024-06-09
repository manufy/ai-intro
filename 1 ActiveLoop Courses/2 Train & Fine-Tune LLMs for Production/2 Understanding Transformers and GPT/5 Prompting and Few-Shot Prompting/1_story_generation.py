import openai

prompt_system = "You are a helpful assistant whose goal is to help write stories."

prompt = """Continue the following story. Write no more than 50 words.

Once upon a time, in a world where animals could speak, a courageous mouse named Benjamin decided to"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[python 1
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0]['message']['content'])