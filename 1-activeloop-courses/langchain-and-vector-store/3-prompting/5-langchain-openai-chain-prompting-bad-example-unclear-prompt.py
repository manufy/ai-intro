# This prompt may generate a less informative or focused response than the previous example
# due to its more open-ended nature.

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Prompt 1
template_question = """What are some musical genres?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Tell me something about {genre1}, {genre2}, and {genre3} without giving any specific details.
Answer: """
prompt_fact = PromptTemplate(input_variables=["genre1", "genre2", "genre3"], template=template_fact)


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

# Assign three hardcoded genres
genre1, genre2, genre3 = "jazz", "pop", "rock"

chain_fact = prompt_fact | llm

# Input data for the second prompt
input_data = {"genre1": genre1, "genre2": genre2, "genre3": genre3}

# Run the LLMChain for the second prompt
response_fact = chain_fact.invoke(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)

# In this example, the second prompt is constructed poorly.
# It asks to "tell me something about {genre1}, {genre2}, and {genre3}
# without giving any specific details." 
# This prompt is unclear, as it asks for information about the genres
# but also states not to provide specific details. This makes it difficult
#  for the LLM to generate a coherent and informative response. 
# As a result, the LLM may provide a less informative or confusing answer.

# The first prompt asks for "some musical genres" without specifying any criteria or context, 
# and the second prompt asks why the given genres are "unique" without providing any guidance
# on what aspects of uniqueness to focus on, such as their historical origins, 
# stylistic features, or cultural significance.