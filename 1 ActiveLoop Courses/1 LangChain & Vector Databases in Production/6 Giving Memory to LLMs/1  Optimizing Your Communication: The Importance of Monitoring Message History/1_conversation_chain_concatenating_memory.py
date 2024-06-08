from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the language model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Initialize the memory
memory = ConversationBufferMemory()

# Initialize the conversation chain with the language model and memory
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Simulate a conversation
output1 = conversation.predict(input="In what scenarios extra memory should be used?")
print("--- Output 1 ---")
print(output1)

# Check memory after first message
print("--- Memory after Output 1 ---")
print(memory.buffer)

output2 = conversation.predict(input="There are various types of memory in Langchain. When to use which type?")
print("--- Output 2 ---")
print(output2)

# Check memory after second message
print("--- Memory after Output 2 ---")
print(memory.buffer)

output3 = conversation.predict(input="Do you remember what was our first message?")
print("--- Output 3 ---")
print(output3)

# Check memory after third message
print("--- Memory after Output 3 ---")
print(memory.buffer)


