cd ..
import cohere
import os

co = cohere.Client(os.environ["COHERE_API_KEY"])

response = co.generate(
    prompt='Please briefly explain to me how Deep Learning works using at most 100 words.',
    max_tokens=200
)
print(response.generations[0].text)