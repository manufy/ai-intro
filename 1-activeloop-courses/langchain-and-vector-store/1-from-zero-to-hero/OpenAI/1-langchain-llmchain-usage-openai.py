from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

#chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
#print(chain.invoke("eco-friendly water bottles"))


product = "eco-friendly water bottles"


runnable_sequence = prompt | llm
result = runnable_sequence.invoke(product)

print(result)