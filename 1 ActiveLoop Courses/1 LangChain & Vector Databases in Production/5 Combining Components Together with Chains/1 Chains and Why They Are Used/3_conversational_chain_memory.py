from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import OpenAI


# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Depending on the application, memory is the next component 
# that will complete a chain. LangChain provides a ConversationalChain 
# to track previous prompts and responses using the ConversationalBufferMemory class.

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

output_parser = CommaSeparatedListOutputParser()
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

result = conversation.predict(input="List all possible words as substitute for 'artificial' as comma separated.")
print(result)

# Now, we can ask it to return the following four replacement words. 
# It uses the memory to find the next options.

result=conversation.predict(input="And the next 4?")
print(result)


