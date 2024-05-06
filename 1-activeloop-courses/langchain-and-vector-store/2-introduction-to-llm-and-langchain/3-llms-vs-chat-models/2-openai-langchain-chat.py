from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage  # Add this line to import AIMessage
)

from termcolor import colored

chat = ChatOpenAI(model_name="gpt-4", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence: I love programming.")
]

print(colored(chat(messages), 'red'))


response = AIMessage(content="J'aime la programmation.", additional_kwargs={}, example=False)


from langchain.schema import LLMResult  # Add this line to import LLMResult

print(colored(response.content, 'yellow'))

from langchain.schema import ChatGeneration  # Add this line to import ChatGeneration

batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate the following sentence: I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates French to English."),
        HumanMessage(content="Translate the following sentence: J'aime la programmation.")
    ],
]


print(colored(chat.generate(batch_messages), 'red'))

response = LLMResult(generations=[[ChatGeneration(text="J'aime la programmation.", generation_info=None, message=AIMessage(content="J'aime la programmation.", additional_kwargs={}, example=False))], [ChatGeneration(text='I love programming.', generation_info=None, message=AIMessage(content='I love programming.', additional_kwargs={}, example=False))]], llm_output={'token_usage': {'prompt_tokens': 65, 'completion_tokens': 11, 'total_tokens': 76}, 'model_name': 'gpt-4'})                                                                                                                   

print(colored(response, 'yellow'))                                                                  