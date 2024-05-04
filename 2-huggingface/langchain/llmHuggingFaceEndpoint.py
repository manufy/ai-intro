from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

import sys
import os

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


stdout_original = sys.stdout
stderr_original = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')


try:
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5
    )
finally:
    # Restaura las salidas estándar y de error originales
    sys.stdout.close()
    sys.stdout = stdout_original
    sys.stderr.close()
    sys.stderr = stderr_original


#llm = HuggingFaceEndpoint(
#    #repo_id=repo_id, max_length=128, temperature=0.5, token= os.getenv("HUGGINGFACEHUB_API_TOKEN")
#    repo_id=repo_id
#)


question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

# for LangCghain < 0.1.17

# llm_chain = LLMChain(prompt=prompt, llm=llm)
# print(llm_chain.invoke(question))

# The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 0.3.0. Use RunnableSequence, e.g., `prompt | llm` instead
# using RunnableSequence for last langchang version should be:

runnable_sequence = prompt | llm
result = runnable_sequence.invoke(question)

print(result)