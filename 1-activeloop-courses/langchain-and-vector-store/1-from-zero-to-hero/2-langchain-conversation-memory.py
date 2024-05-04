
# Quick intro to llm invoke call , as shown in the course
# For speed reasons I do not to use a local model, neither OpenAI to avoid running costs

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


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

conversation = ConversationChain(
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation.memory.buffer)
