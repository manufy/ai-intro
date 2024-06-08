from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)


prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)


print( conversation.predict(input="Tell me a joke about elephants") )
print( conversation.predict(input="Who is the author of the Harry Potter series?") )
print( conversation.predict(input="What was the joke you told me earlier?") )