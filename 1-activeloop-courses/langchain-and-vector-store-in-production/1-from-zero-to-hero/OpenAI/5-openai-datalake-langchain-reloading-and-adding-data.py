from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = os.environ["ACTIVELOOP_DATASET_OPENAI"]
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# load the existing Deep Lake dataset and specify the embedding function
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# Deep Lake Dataset in hub://manuai/langchain_course_from_zero_to_hero_openai already exists, loading from the storage
# Creating 2 embeddings in 1 batches of size 2:: 100%|█| 1/1 [00:33<00:00, 33.62s
# Dataset(path='hub://manuai/langchain_course_from_zero_to_hero_openai', tensors=['embedding', 'id', 'metadata', 'text'])

#   tensor      htype      shape     dtype  compression
#   -------    -------    -------   -------  ------- 
#  embedding  embedding  (4, 1536)  float32   None   
#     id        text      (4, 1)      str     None   
#  metadata     json      (4, 1)      str     None   
#    text       text      (4, 1)      str     None   