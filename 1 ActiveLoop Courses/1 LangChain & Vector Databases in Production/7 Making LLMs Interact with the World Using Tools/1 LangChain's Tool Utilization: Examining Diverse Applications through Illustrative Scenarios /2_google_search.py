import os



# As a standalone utility:
from langchain_google_community import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
result = search.results("What is the capital of Spain?", 3)
print(result)


from langchain_openai import  OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)




from langchain.agents import initialize_agent, load_tools, AgentType

tools = load_tools(["google-search"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

print( agent("What is the national drink in Spain?") )