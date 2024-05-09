from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Deprecated:
# chain = LLMChain(llm=llm, prompt=prompt)

chain = prompt | llm

response = chain.invoke({"question":"what is the meaning of life?", "llm": llm, "prompt": prompt})

from termcolor import colored

print(colored(response, 'yellow'))