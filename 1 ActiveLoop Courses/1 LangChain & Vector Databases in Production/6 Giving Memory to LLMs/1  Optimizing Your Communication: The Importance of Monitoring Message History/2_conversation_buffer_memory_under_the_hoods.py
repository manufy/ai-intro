from langchain_openai import OpenAI
from langchain.chains import ConversationChain

from langchain.memory import ConversationBufferMemory

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())


memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "hi there!"}, {"output": "Hi there! It's nice to meet you. How can I help you today?"})

print( memory.load_memory_variables({}) )