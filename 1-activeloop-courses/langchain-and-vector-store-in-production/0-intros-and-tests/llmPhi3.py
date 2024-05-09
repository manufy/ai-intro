from langchain.llms import OpenAI
from langchain_community.llms import Ollama

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
#llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
llm = Ollama(model="phi3")
#text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
text = "Tell me a Joke"
print("---- plain ----")
a=llm.invoke(text)
print(a)
print("---- stream ----")

for chunks in llm.stream(text):
    print(chunks)
print("---- curso ----")
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))