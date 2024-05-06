
from langchain_community.llms import HuggingFaceEndpoint

# initialize Hub LLM
model = "mistralai/Mistral-7B-Instruct-v0.2"

# Using google model need community to update huggingface_models.py
# model = "google/flan-t5-large"
hub_llm = HuggingFaceEndpoint(
        repo_id=model,
        temperature = 0.1, # google accepts temperature 0, but in mistral should be positive
        max_new_tokens = 100
)


model = "mistralai/Mistral-7B-Instruct-v0.2"

from langchain_core.prompts import PromptTemplate

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""

long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
		"What color is a ripe banana?\n"
)

runnable_sequence = long_prompt | hub_llm

result = runnable_sequence.invoke(qs_str)

from termcolor import colored

print(colored(result, 'yellow'))
