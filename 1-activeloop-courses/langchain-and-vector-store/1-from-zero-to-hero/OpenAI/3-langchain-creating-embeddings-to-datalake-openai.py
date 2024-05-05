from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import os

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

# instantiate the LLM and embeddings models
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = os.environ["ACTIVELOOP_DATASET_OPENAI"]
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)