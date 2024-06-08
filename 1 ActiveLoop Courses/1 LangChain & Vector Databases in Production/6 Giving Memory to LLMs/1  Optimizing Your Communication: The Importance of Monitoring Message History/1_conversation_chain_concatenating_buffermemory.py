#from langchain import OpenAI, ConversationChain
from langchain_openai import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "hi there!"}, {"output": "Hi there! It's nice to meet you. How can I help you today?"})

print( memory.load_memory_variables({}) )

output = conversation.predict(input="In what scenarios extra memory should be used?")
output = conversation.predict(input="There are various types of memory in Langchain. When to use which type?")
output = conversation.predict(input="Do you remember what was our first message?")


print(output)
print("------------------")
print( memory.load_memory_variables({}) )
