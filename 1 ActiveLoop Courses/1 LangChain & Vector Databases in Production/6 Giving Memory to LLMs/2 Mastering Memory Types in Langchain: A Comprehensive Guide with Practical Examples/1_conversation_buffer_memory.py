#from langchain import OpenAI, ConversationChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# TODO: Set your OPENAI API credentials in environemnt variables.
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)
result = conversation.predict(input="Hello!")
print(result)