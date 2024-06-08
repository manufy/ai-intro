from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAI

from langchain.load.dump import dumps

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)

user_message = "Tell me about the history of the Internet."
response = conversation(user_message)

import pprint
print("------------")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(response)



# User sends another message
user_message = "Who are some important figures in its development?"
response = conversation(user_message)
pp.pprint(response)  # Chatbot responds with names of important figures, recalling the previous topic
    
print("------------")
    
user_message = "What did Tim Berners-Lee contribute?"
response = conversation(user_message)
print(response)
print ("---------   ")

print(response['response'])