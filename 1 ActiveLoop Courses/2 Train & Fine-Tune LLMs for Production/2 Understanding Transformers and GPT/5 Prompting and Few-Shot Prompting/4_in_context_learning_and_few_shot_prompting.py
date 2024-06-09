import openai

prompt_system = "You are a helpful assistant whose goal is to write short poems."

prompt = """Write a short poem about {topic}."""

examples = {
    "nature": "Birdsong fills the air,\nMountains high and valleys deep,\nNature's music sweet.",
    "winter": "Snow blankets the ground,\nSilence is the only sound,\nWinter's beauty found."
}

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt.format(topic="nature")},
        {"role": "assistant", "content": examples["nature"]},
        {"role": "user", "content": prompt.format(topic="winter")},
        {"role": "assistant", "content": examples["winter"]},
        {"role": "user", "content": prompt.format(topic="summer")}
    ]
)

print(response.choices[0]['message']['content']) 