import openai

prompt_system = "You are a helpful assistant whose goal is to write short poems."

prompt = """Write a short poem about {topic}."""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt.format(topic="summer")}
    ]
)

print(response.choices[0]['message']['content'])