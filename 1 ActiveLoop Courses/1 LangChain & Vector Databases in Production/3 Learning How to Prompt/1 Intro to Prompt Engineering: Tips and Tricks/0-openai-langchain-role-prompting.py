
# In this example, the LLM is asked to act as a futuristic robot band conductor 
# and suggest a song title related to the given theme and year.

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""
prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

# Create the LLMChain for the prompt
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Input data for the prompt
input_data = {"theme": "interstellar travel", "year": "3030"}

# Create LLMChain
chain = prompt | llm

from termcolor import colored

# Run the LLMChain to get the AI-generated song title
response = chain.invoke(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print(colored("AI-generated song title:", 'green'), colored(response, 'yellow'))
