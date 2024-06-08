from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, verbose=True)
tools = load_tools(['serpapi', 'requests_all'], llm=llm, serpapi_api_key="<YOUR_SERPAPI_API_KEY>")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True)


text = agent.run("tell me what is midjourney?")

print(text)
