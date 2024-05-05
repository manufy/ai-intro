from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import json
from termcolor import colored


# pip install google-api-python-client
# API KEY and CUSTOM SEARCH creation https://console.cloud.google.com/apis/dashboard
# API KEY: https://console.cloud.google.com/apis/credentials
# CSE_ID https://programmablesearchengine.google.com/controlpanel/overview

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)


print(colored("--- Google search Obama's first name ---", 'green'))
print(colored(tool.run("Obama's first name?"), 'yellow'))


tool = Tool(
    name="I'm Feeling Lucky",
    description="Search Google and return the first result.",
    func=search.run,
)

print(colored("--- Google search for python, one result ts ---", 'green'))
print(colored(tool.run("python"), 'yellow'))

def top5_results(query):
    return search.results(query, 5)


tool = Tool(
    name="Google Search Snippets",
    description="Search Google for recent results.",
    func=top5_results,
)


print(colored("--- Google search for python, five top results ---", 'green'))
print(colored(json.dumps(tool.run("python"), indent=4), 'yellow'))