from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(temperature=0)

prompt = PromptTemplate(
  input_variables=["product"],
  template="What is a good name for a company that makes {product}?",
)

# Deprecated:
# chain = LLMChain(llm=llm, prompt=prompt)

chain = prompt | llm

response = chain.invoke({"product":"wireless headphones", "llm": llm, "prompt": prompt})

from termcolor import colored

print(colored(response, 'yellow'))