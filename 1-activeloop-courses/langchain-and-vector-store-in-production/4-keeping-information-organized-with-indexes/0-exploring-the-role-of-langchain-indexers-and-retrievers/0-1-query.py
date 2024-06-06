import os
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]

my_activeloop_dataset_name = "indexers-retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)


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
# response = qa_chain.run(query)
response = qa_chain.invoke(query)


print(response)