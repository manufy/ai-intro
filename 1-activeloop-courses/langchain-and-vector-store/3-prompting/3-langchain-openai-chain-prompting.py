from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# Create the LLMChain for the first prompt
#chain_question = LLMChain(llm=llm, prompt=prompt_question)

chain_question = prompt_question | llm

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.invoke({})

print ("Response question:", response_question)

# Extract the scientist's name from the response
scientist = response_question.strip()

# Create the LLMChain for the second prompt
# chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

chain_fact = prompt_fact | llm

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.invoke(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)