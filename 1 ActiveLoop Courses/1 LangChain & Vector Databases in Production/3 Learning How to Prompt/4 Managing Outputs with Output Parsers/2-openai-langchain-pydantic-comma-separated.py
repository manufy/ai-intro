from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()






from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

# Prepare the Prompt
template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
  target_word="behaviour",
  context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Loading OpenAI API
model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0)

# Send the Request
output = model(model_input)
result = parser.parse(output)
print(output)
print ("---- Result ----")
print(result)