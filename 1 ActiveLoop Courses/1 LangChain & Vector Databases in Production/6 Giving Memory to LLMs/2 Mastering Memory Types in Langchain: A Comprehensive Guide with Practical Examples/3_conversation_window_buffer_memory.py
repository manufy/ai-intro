from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import PromptTemplate

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

template = """You are ArtVenture, a cutting-edge virtual tour guide for
 an art gallery that showcases masterpieces from alternate dimensions and
 timelines. Your advanced AI capabilities allow you to perceive and understand
 the intricacies of each artwork, as well as their origins and significance in
 their respective dimensions. As visitors embark on their journey with you
 through the gallery, you weave enthralling tales about the alternate histories
 and cultures that gave birth to these otherworldly creations.

{chat_history}
Visitor: {visitor_input}
Tour Guide:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "visitor_input"], 
    template=template
)

chat_history=""

convo_buffer_win = ConversationChain(
    llm=llm,
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
)

convo_buffer_win("What is your name?")
convo_buffer_win("What can you do?")
convo_buffer_win("Do you mind give me a tour, I want to see your galery?")
convo_buffer_win("what is your working hours?")
result = convo_buffer_win("See you soon.")
print(result)