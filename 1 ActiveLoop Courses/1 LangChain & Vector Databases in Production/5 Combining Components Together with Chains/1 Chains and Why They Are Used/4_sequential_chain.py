
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import OpenAI


# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)



# Another helpful feature is using a sequential chain that concatenates 
# multiple chains into one. The following code shows a sample usage.


# poet
poet_template: str = """You are an American poet, your job is to come up with\
poems based on a given theme.

Here is the theme you have been asked to generate a poem on:
{input}\
"""

poet_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["input"], template=poet_template)

# creating the poet chain
poet_chain: LLMChain = LLMChain(
    llm=llm, output_key="poem", prompt=poet_prompt_template)

# critic
critic_template: str = """You are a critic of poems, you are tasked\
to inspect the themes of poems. Identify whether the poem includes romantic expressions or descriptions of nature.

Your response should be in the following format, as a Python Dictionary.
poem: this should be the poem you received 
Romantic_expressions: True or False
Nature_descriptions: True or False

Here is the poem submitted to you:
{poem}\
"""

critic_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["poem"], template=critic_template)

# creating the critic chain
#critic_chain: LLMChain = LLMChain(
#    llm=llm, output_key="critic_verified", prompt=critic_prompt_template)

critic_chain = critic_prompt_template | llm 
result = critic_chain.invoke(critic_chain("The sun is shining bright"))

print(critic_chain)
print(result)