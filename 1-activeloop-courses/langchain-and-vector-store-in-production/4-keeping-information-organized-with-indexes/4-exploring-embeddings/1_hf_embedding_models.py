#from langchain.llms import HuggingFacePipeline
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.llms import HuggingFacePipeline
#from langchain_community.llms import HuggingFacePipeline

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",cache_folder="./custom_cache")

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
# hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
#hf = HuggingFacePipeline(model_name=model_name, model_kwargs=model_kwargs)

documents = ["Document 1", "Document 2", "Document 3"]
doc_embeddings = model.encode(documents)
#doc_embeddings = model.embed_documents(documents)
print(doc_embeddings)
