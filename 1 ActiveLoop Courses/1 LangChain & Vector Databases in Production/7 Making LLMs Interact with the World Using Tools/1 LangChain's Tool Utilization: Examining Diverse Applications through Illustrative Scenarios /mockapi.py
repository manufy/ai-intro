from langchain.agents import AgentType

from langchain_openai import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, verbose=True)

tools = load_tools(["requests_all"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent="chat-zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
  )
response = agent.run("Get the list of users at https://644696c1ee791e1e2903b0bb.mockapi.io/user and tell me the total number of users")