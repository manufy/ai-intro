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
input_messages = [
    "In what scenarios extra memory should be used?",
    "There are various types of memory in Langchain. When to use which type?",
    "Do you remember what was our first message?"
]
output_messages = []

# Iterate over input messages
for input_message in input_messages:
    # Predict response
    output = conversation.predict(input=input_message)
    print("--- Output ---")
    print(output)
    output_messages.append(output)

    # Ensure memory buffer is initialized as a list
    if not isinstance(memory.buffer, list):
        memory.buffer = []
    
    # Add input-output pair to memory buffer
    memory.buffer.append((input_message, output))

# Check memory after the conversation
print("--- Memory ---")
print(memory.buffer)
