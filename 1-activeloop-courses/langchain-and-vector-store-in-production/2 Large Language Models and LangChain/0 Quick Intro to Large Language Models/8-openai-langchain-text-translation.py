from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)




translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"], template=translation_template)

runnable_sequence = translation_prompt | llm

source_language = "English"
target_language = "French"
text = "Your text here"

summarized_text = runnable_sequence.invoke({"text": text, "source_language": source_language, "target_language": target_language})

from termcolor import colored

print(colored(summarized_text.content, 'yellow'))