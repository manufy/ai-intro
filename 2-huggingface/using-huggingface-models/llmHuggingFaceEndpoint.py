from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

import os

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# Suprime las advertencias de UserWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )

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

runnable_sequence = prompt | llm

# Invoke the sequence with the question
result = runnable_sequence.invoke(question)

print(result)