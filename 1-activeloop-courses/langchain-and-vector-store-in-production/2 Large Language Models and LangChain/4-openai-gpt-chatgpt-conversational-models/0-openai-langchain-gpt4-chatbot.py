from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

# The code below adds a new message which requires the model
# to understand and find the “city you just mentioned” reference from previous conversations.

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)
# add to messages
messages.append(prompt)

llm = ChatOpenAI(model_name="gpt-4")

response = llm(messages)
from termcolor import colored

print(colored(response.content, 'yellow'))