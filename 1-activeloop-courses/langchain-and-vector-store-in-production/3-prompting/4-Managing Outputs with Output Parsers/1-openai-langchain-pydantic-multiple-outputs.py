from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.output_parsers import PydanticOutputParser

class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")
    
    @field_validator('words')
    def not_start_with_number(cls, field):
      for item in field:
        if item[0].isnumeric():
          raise ValueError("The word can not start with numbers!")
      return field
    
    @field_validator('reasons')
    def end_with_dot(cls, field):
      for idx, item in enumerate( field ):
        if item[-1] != ".":
          field[idx] += "."
      return field



parser = PydanticOutputParser(pydantic_object=Suggestions)


from langchain.prompts import PromptTemplate

template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
			target_word="behaviour",
			context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)









from langchain_openai import OpenAI

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0)

output = model(model_input.to_string())

parser.parse(output)




result = Suggestions(words=['conduct', 'manner', 'demeanor', 'comportment'], reasons=['refers to the way someone acts in a particular situation.', 'refers to the way someone behaves in a particular situation.', 'refers to the way someone behaves in a particular situation.', 'refers to the way someone behaves in a particular situation.'])
print(output)