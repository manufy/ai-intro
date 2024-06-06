
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import OpenAI


# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

output_parser = CommaSeparatedListOutputParser()
template = """List all possible words as substitute for 'artificial' as comma separated."""

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=template, output_parser=output_parser, input_variables=[]),
    output_parser=output_parser)

result = llm_chain.predict()
print(result)