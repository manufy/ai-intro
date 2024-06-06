from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import os


llm_model = "google-bert/bert-base-uncased"

# instantiate the LLM and embeddings models
llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        temperature=0.5
    )
embeddings = HuggingFaceEmbeddings(model_name=llm_model)

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
my_activeloop_dataset_name = os.environ["ACTIVELOOP_DATASET"]
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)