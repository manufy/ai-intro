from langchain import OpenAI, ConversationChain
#from langchain_openai import OpenAI
#from langchain.chains import ConversationChain

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="In what scenarios extra memory should be used?")
output = conversation.predict(input="There are various types of memory in Langchain. When to use which type?")
output = conversation.predict(input="Do you remember what was our first message?")

print ("---- Output ----")  
print(output)
