# Import necessary modules
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

# pip install pypdf

# Initialize language model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="whatisai.pdf")
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
from termcolor import colored

print(colored(summary['output_text'], 'yellow'))
