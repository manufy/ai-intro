from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

parser.parse(missformatted_output)

# above code shows 

 #   raise self._parser_exception(e, obj)
# langchain_core.exceptions.OutputParserException: Failed to parse Suggestions from completion {"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}. Got: 1 validation error for Suggestions
# reasons
#   Field required [type=missing, input_value={'words': ['conduct', 'ma...particular situation.']}, input_type=dict]
#    For further information visit https://errors.pydantic.dev/2.7/v/missing

# the fix is 

from langchain_community.llms import OpenAI
from langchain.output_parsers import OutputFixingParser

model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0)

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
outputfixing_parser.parse(missformatted_output)