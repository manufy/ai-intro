# Quick intro to llm invoke call , as shown in the course
# For speed reasons I do not to use a local model, neither OpenAI to avoid running costs

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

import sys
import os

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities. only show a short activity description no longer than 20 characters and estimated time."
print(llm.invoke(text))