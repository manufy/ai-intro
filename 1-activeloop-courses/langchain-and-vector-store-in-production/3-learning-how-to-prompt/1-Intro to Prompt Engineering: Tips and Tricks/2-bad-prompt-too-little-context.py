
#from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#from langchain.llms import OpenAI
from langchain_openai import OpenAI

template = "Tell me something about {topic}."
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)
prompt.format(topic="dogs")

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

chain = prompt | llm

response = chain.invoke({"topic": "dogs"})
print(response)