from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API. https://console.cloud.google.com/apis/api/customsearch.googleapis.com
# https://api.python.langchain.com/en/latest/utilities/langchain_community.utilities.google_search.GoogleSearchAPIWrapper.html

search = GoogleSearchAPIWrapper()