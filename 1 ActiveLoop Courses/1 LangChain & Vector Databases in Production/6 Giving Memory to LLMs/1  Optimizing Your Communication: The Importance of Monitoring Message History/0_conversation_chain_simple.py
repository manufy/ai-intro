
#from langchain import OpenAI, ConversationChain
from langchain_openai import OpenAI
from langchain.chains import ConversationChain


llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")

print(output)