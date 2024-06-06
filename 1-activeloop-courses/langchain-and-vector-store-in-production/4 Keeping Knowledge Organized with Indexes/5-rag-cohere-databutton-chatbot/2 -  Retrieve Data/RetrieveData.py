from langchain.vectorstores import DeepLake
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import os

os.environ["COHERE_API_KEY"] = db.secrets.get("COHERE_API_KEY")
os.environ["ACTIVELOOP_TOKEN"] = db.secrets.get("ACTIVELOOP_TOKEN")

@st.cache_resource()
def data_lake():
    embeddings = CohereEmbeddings(model = "embed-english-v2.0")

    dbs = DeepLake(
        dataset_path="hub://elleneal/activeloop-material", 
        read_only=True, 
        embedding_function=embeddings
        )
    retriever = dbs.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    compressor = CohereRerank(
        model = 'rerank-english-v2.0',
        top_n=5
        )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    return dbs, compression_retriever, retriever

dbs, compression_retriever, retriever = data_lake()
