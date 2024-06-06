from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI


prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

result = llm_chain("artificial")
print(result)

# It is also possible to use the .apply() method to pass multiple inputs
# at once and receive a list for each input. 
# The sole difference lies in the exclusion of inputs within the returned list.
# Nonetheless, the returned list will maintain the identical order as the input.

input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]

result = llm_chain.apply(input_list)
print(result)

# The .generate() method will return an instance of LLMResult, 
# hich provides more information. 
# For example, the finish_reason key indicates the reason
# behind the stop of the generation process. 
# It could be stopped, meaning the model decided 
# to finish or reach the length limit. 
# There is other self-explanatory information 
# like the number of total used tokens or the used model.

result  = llm_chain.generate(input_list)
print(result)

# The next method we will discuss is .predict(). (which could be used interchangeably with .run())
# Its best use case is to pass multiple inputs for a single prompt. 
# However, it is possible to use it with one input variable as well.
# The following prompt will pass both the word we want a substitute 
# for and the context the model must consider.

prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["word", "context"]))

result = llm_chain.predict(word="fan", context="object")
# or llm_chain.run(word="fan", context="object")
print(result)

# The model correctly suggested that a Ventilator would be a suitable replacement 
# for the word fan in the context of objects. Furthermore,
# when we repeat the experiment with a different context, humans,
# the output will change the Admirer.

result = llm_chain.predict(word="fan", context="humans")
# or llm_chain.run(word="fan", context="object")
print(result)
