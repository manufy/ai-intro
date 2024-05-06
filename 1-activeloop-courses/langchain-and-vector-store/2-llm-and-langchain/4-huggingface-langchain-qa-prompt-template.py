from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "What is the capital city of France?"

# Next, we will use the Hugging Face model google/flan-t5-large to answer the question. 
# The HuggingfaceHub class will connect to Hugging Face’s inference API and load the specified model.

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain

# initialize Hub LLM
model = "mistralai/Mistral-7B-Instruct-v0.2"

# Using google model need community to update huggingface_models.py
# model = "google/flan-t5-large"
hub_llm = HuggingFaceEndpoint(
        repo_id=model,
        temperature = 0.1, # google accepts temperature 0, but in mistral should be positive
        max_new_tokens = 100
)

runnable_sequence = prompt | hub_llm

# Invoke the sequence with the question
# ask the user question about the capital of France
result = runnable_sequence.invoke(question)
from termcolor import colored
# ask the user question about the capital of France
print(colored(result, "yellow"))