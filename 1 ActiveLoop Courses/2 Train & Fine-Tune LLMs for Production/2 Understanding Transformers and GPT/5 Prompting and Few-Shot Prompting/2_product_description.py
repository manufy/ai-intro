import openai

prompt_system = "You are a helpful assistant whose goal is to help write product descriptions."

prompt = """Write a captivating product description for a luxurious, hand-crafted, limited-edition fountain pen made from rosewood and gold.
Write no more than 50 words."""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0]['message']['content'])