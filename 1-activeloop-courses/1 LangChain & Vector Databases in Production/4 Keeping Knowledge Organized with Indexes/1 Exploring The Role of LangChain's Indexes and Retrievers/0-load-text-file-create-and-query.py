# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

# Here we use the TextLoader class to load a text file. 
# Remember to install the required packages with the following command:
# pip install langchain==0.0.208 deeplake openai==0.27.8 tiktoken.


# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# write text to local file
with open("my_file.txt", "w") as file:
    file.write(text)

# use TextLoader to load text from local file
loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

print(len(docs_from_file))
# 1

# 
# Then, we use CharacterTextSplitter to split the docs into texts.

from langchain.text_splitter import CharacterTextSplitter

# create a text splitter
#text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

text_splitter = CharacterTextSplitter(chunk_size=373, chunk_overlap=20)

# split documents into chunks
docs = text_splitter.split_documents(docs_from_file)

print(len(docs))
# 2

# These embeddings allow us to effectively search for documents or portions of documents
# that relate to our query by examining their semantic similarities. 

# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Let’s create an instance of a Deep Lake dataset.

#from langchain.vectorstores import DeepLake
from langchain_community.vectorstores import DeepLake

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
import os
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]

my_activeloop_dataset_name = "indexers-retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
# add documents to our Deep Lake dataset
db.add_documents(docs)


# In this example, we are adding text documents to the dataset.
# However, being Deep Lake multimodal, we could have also added images to it, 
# specifying an image embedder model. This could be useful for searching images 
# according to a text query or using an image as a query (and thus looking for similar images).

# As datasets become bigger, storing them in local memory becomes less manageable. 
# In this example, we could have also used a local vector store, as we are uploading only two documents.
# However, in a typical production scenario, thousands or millions of documents 
# could be used and accessed from different programs, 
# thus having the need for a centralized cloud dataset.

# Back to the code example of this lesson. Next, we create a retriever.

# create retriever from db
retriever = db.as_retriever()

# Once we have the retriever, we can start with question-answering.

from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
from langchain_openai import OpenAI

# create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
	llm=OpenAI(model="gpt-3.5-turbo-instruct"),
	chain_type="stuff",
	retriever=retriever
)


# We can query our document that is an about specific topic that can be found in the documents.

query = "How Google plans to challenge OpenAI?"
response = qa_chain.run(query)
print(response)