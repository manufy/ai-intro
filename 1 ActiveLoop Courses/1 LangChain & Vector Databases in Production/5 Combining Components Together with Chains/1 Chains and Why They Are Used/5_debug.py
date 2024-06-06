
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser


output_parser = CommaSeparatedListOutputParser()

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)


template = """List all possible words as substitute for 'artificial' as comma separated.

Current conversation:
{history}

{input}"""

conversation = ConversationChain(
    llm=llm,
    prompt=PromptTemplate(template=template, input_variables=["history", "input"], output_parser=output_parser),
    memory=ConversationBufferMemory(),
    verbose=True)

result = conversation.predict(input="")