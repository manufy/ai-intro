import os
from termcolor import colored
print(colored("----- Importing dependencies -----", 'yellow'))
from deeplake_functions import get_connection, get_retriever
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

print(colored("----- Establishing DeepLake binding -----", 'yellow'))
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
deeplake_db, retriever = get_retriever(os.environ["ACTIVELOOP_ORG_ID"], "indexers-retrievers", embeddings)

print(colored("----- Retrieving document -----", 'yellow'))

# create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
	llm=OpenAI(model="gpt-3.5-turbo-instruct"),
	chain_type="stuff",
	retriever=retriever
)

# We can query our document that is an about specific topic that can be found in the documents.

query = "How Google plans to challenge OpenAI?"
response = qa_chain.invoke(query)


print(response)

