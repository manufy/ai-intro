
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
# from langchain_openai import OpenAI

#from langchain.chains import LLMChain

# Regarding This is a chat model and not supported in the v1/completions endpoint error

# The code you posted above would work immediately
#  if you changed just one thing: gpt-3.5-turbo to text-davinci-003. 
# This gives you an answer as to why you're getting this error.
# It's because you used the code that works with the GPT-3 API endpoint,
#  but wanted to use the GPT-3.5 model (i.e., gpt-3.5-turbo).
#  See model endpoint compatibility.

# https://stackoverflow.com/questions/75774873/openai-api-error-this-is-a-chat-model-and-not-supported-in-the-v1-completions

# at the end is solved with: 

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"}
]

example_template = """
Animal: {animal}
Habitat: {habitat}
"""

example_prompt = PromptTemplate(
    input_variables=["animal", "habitat"],
    template=example_template
)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal: {input}\nHabitat:",
    input_variables=["input"],
    example_separator="\n\n",
)

# Create the LLMChain for the dynamic_prompt
#chain = LLMChain(llm=llm, prompt=dynamic_prompt)

chain = dynamic_prompt | llm

# Run the LLMChain with input_data
input_data = {"input": "tiger"}
response = chain.invoke(input_data)

print(response.content)

# save and load prompt

example_prompt.save("awesome_prompt.json")

from langchain.prompts import load_prompt
loaded_prompt = load_prompt("awesome_prompt.json")
