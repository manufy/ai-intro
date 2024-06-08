from deeplake import VectorStore

from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain    
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

corpus = VectorStore(
    "hub://activeloop-test/scifact-demo",
    embedding_function = embeddings.embed_documents,
    runtime={"tensor_db": True}
)





questions = ["question 1", ...]
relevance = [[(corpus.dataset.id[0], 1), ...], ...]

job_id = corpus.deep_memory.train(
    queries = questions,
    relevance = relevance,
    embedding_function = embeddings.embed_documents,
)

corpus.deep_memory.status(job_id)



corpus.search(
    embedding_data = "Female carriers of the Apolipoprotein E4 (APOE4) allele have increased risk for dementia.",
    embedding_function = embeddings.embed_query,
    deep_memory = True
)


corpus.deep_memory.evaluate(
    queries = test_questions,
    relevance = test_relevance,
    embedding_function = embeddings.embed_documents,
    top_k=[1, 3, 5, 10, 50, 100]
)





