from langchain.output_parsers import PydanticOutputParser
#from pydantic import validator
from pydantic import BaseModel, Field, field_validator
#from pydantic import BaseModel, Field
from typing import List

from fetch_article_function import get_article

article = get_article()
article = get_article()
article_title = article.title
article_text  = article.text


# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    #@field_validator('summary', allow_reuse=True)
    @field_validator('summary')
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

# The next step involves creating a template for the input prompt
# that instructs the language model to summarize the news article into bullet points. 
# This template is used to instantiate a PromptTemplate object, which is responsible 
# for correctly formatting the prompts that are sent to the language model. 
# The PromptTemplate uses our custom parser to format the prompt sent to the language model
# using the .get_format_instructions() method, which will include additional instructions
# on how the output should be shaped.

from langchain.prompts import PromptTemplate


# create prompt template
# notice that we are specifying the "partial_variables" parameter
template = """
You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Format the prompt using the article title and text obtained from scraping
formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)

# Lastly, the GPT-3 model with the temperature set to 0.0  is initialized
# which means the output will be deterministic, 
# favoring the most likely outcome over randomness/creativity. 
# The parser object then converts the string output from the model to 
# a defined schema using the .parse() method.

#from langchain.llms import OpenAI
from langchain_openai import OpenAI
# instantiate model class
model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

# Use the model to generate a summary
output = model(formatted_prompt.to_string())

# Parse the output into the Pydantic model
parsed_output = parser.parse(output.split("\"]}")[0] + "\"]}")
print(parsed_output)

ArticleSummary(title='Meta claims its new AI supercomputer will set records', summary=['Meta (formerly Facebook) has unveiled an AI supercomputer that it claims will be the world’s fastest.', 'The supercomputer is called the AI Research SuperCluster (RSC) and is yet to be fully complete.', 'Meta says that it will be the fastest in the world once complete and the aim is for it to be capable of training models with trillions of parameters.', 'For production, Meta expects RSC will be 20x faster than Meta’s current V100-based clusters.', 'Meta says that its previous AI research infrastructure only leveraged open source and other publicly-available datasets.', 'What this means in practice is that Meta can use RSC to advance research for vital tasks such as identifying harmful content on its platforms—using real data from them.'])