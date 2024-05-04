# Quick intro to llm invoke call , as shown in the course
# For speed reasons I do not to use a local model, neither OpenAI to avoid running costs

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


import sys
import os

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)


# TODO: find a better way to handle stdout and stderr message suppression

stdout_original = sys.stdout
stderr_original = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Initialize the HuggingFace endpoint to use with mistral model

try:
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5
    )
finally:
    sys.stdout.close()
    sys.stdout = stdout_original
    sys.stderr.close()
    sys.stderr = stderr_original

# This call will show a list of activities and estimated time
# the course promt was ""Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
# a litlle bit of context was added to the prompt to make mistral output similar to openai output shown in the course
# Original OpenAI prompt was: "What is a good name for a company that makes {product}?

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}? give me only the name of the company with no more explanations. Give only 1 result.",
)

print ("---- Output with LLMChain ----")
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.invoke("eco-friendly water bottles")['text'])



print ("---- Output with RunnableSequence ----")

runnable_sequence = prompt | llm

# Invoke the sequence with the question
result = runnable_sequence.invoke(prompt)

print(result)