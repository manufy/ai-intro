from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

summarization_template = "Summarize the following text to one sentence: {text}"

summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)

# Deprecated: LLMChain
# summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
# summarized_text = summarization_chain.predict(text=text)

runnable_sequence = summarization_prompt | llm

text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."
summarized_text = runnable_sequence.invoke({"text": text})

from termcolor import colored

print(colored(summarized_text.content, 'yellow'))