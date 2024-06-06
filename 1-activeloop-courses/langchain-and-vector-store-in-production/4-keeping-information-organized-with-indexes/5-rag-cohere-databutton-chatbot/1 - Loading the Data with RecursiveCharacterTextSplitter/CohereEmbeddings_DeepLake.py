from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
import os

os.environ["COHERE_API_KEY"] = db.secrets.get("COHERE_API_KEY")
os.environ["ACTIVELOOP_TOKEN"] = db.secrets.get("APIFY_API_TOKEN")

embeddings = CohereEmbeddings(model = "embed-english-v2.0")

username = "elleneal" # replace with your username from app.activeloop.ai
db_id = 'kb-material'# replace with your database name
DeepLake.force_delete_by_path(f"hub://{username}/{db_id}")

dbs = DeepLake(dataset_path=f"hub://{username}/{db_id}", embedding_function=embeddings)
dbs.add_documents(docs_split)
