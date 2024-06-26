# Importing necessary modules
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# Loading the language model to control the agent
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# Loading some tools to use. The llm-math tool uses an LLM, so we pass that in.
tools = load_tools(["google-search", "llm-math"], llm=llm)

# Initializing an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Testing the agent
query = "What's the result of 1000 plus the number of goals scored in the soccer world cup in 2018?"
response = agent.run(query)
print(response)