import warnings
warnings.filterwarnings("ignore")
from langchain._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


from langchain_openai import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent


from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import json
from termcolor import colored


# to solve deprecations:
# from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
# from langchain_community.agents import create_react_agent

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API. https://console.cloud.google.com/apis/api/customsearch.googleapis.com
# https://api.python.langchain.com/en/latest/utilities/langchain_community.utilities.google_search.GoogleSearchAPIWrapper.html

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

# Deprecated:
agent = initialize_agent(tools, 
                        llm, 
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True,
                        max_iterations=6)

response = agent("What's the latest news about the Mars rover?")
print(response['output'])

# using pip install langchain-community
#agent = create_react_agent(tools, 
#                           llm, 
#                           verbose=True,
#                           max_iterations=6)

#response = agent.invoke("What's the latest news about the Mars rover?")
#print(response['output'])