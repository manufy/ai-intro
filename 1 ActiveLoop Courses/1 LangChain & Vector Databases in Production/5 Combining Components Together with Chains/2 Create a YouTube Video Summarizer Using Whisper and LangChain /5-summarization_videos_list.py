from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

# Load the texts
with open('text.txt') as f:
    text = f.read()
texts = text_splitter.split_text(text)

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
texts = text_splitter.split_text(text)

# Pack all chunks into a documents

from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]

# import Deep Lake and build a database with embedded documents:

from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
import os
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)

# use a retriever to get the documents

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

#  The distance metric determines how the Retriever measures "distance" or similarity
# between different data points in the database.
# By setting distance_metric to 'cos', the Retriever will use
# cosine similarity as its distance metric. Cosine similarity is a
# measure of similarity between two non-zero vectors of an inner product space
# that measures the cosine of the angle between them.
# It's often used in information retrieval to measure the similarity
# between documents or pieces of text. Also, by setting 'k' to 4, 
# the Retriever will return the 4 most similar or closest results
# according to the distance metric when a search is performed.

# We can construct and use a custom prompt template with the QA chain. 
# The RetrievalQA chain is useful to query similiar contents from 
# databse and use the returned records as context to answer questions. 
# The custom prompt ability gives us the flexibility to define 
# custom tasks like retrieving the documents and summaizing the results
# in a bullet-point style.

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullter points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# Lastly, we can use the chain_type_kwargs argument to define the 
# custom prompt and for chain type the ‘stuff’  variation was picked.
# You can perform and test other types as well, as seen previously.

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)

print( qa.run("Summarize the mentions of google according to their AI program") )