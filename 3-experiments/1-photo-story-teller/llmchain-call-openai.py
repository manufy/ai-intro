# import
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
load_dotenv(find_dotenv())
# load OpenAI LLM model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
# Let LLM generate for one input
llm("Tell me a joke")
llm_results = llm.generate(["Tell me a joke", "Write me a song"])
print(llm_results.generations)