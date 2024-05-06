
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

qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]

from langchain_core.prompts import PromptTemplate


template = """Question: {question}

Answer: """

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# deprecated: from langchain.chains import LLMChain
# create prompt template > LLM chain

# from langchain.chains import LLMChain
# llm_chain = LLMChain(
#    prompt=prompt,
#    llm=hub_llm
# )

# ask the user question about the capital of France
# print(llm_chain.run(question))


runnable_sequence = prompt | hub_llm

# user question
question = "What is the capital city of France?"

result = runnable_sequence.invoke(qa)
from termcolor import colored

print(colored(result, 'yellow'))

