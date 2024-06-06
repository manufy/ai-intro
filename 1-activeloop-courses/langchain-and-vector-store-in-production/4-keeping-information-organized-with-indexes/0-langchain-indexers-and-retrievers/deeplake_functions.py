
import os
from termcolor import colored

print(colored("----- Importing DeepLake -----", 'yellow'))
from langchain_community.vectorstores import DeepLake

def get_connection(org_id, dataset_name, embeddings):
    print(colored("----- Connecting to DeepLake -----", 'yellow'))
    activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
    my_activeloop_dataset_name = "indexers-retrievers"
    dataset_path = f"hub://{activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
    return db

def get_retriever(org_id, dataset_name, embeddings, deeplake_db=None,):
    print(colored("----- Obtaining DeepLake retriever -----", 'yellow'))
    # Check if we have connection opened
    if deeplake_db==None:
        print(colored("----- no DB set, obtaining it -----", 'yellow'))
        deeplake_db = get_connection(org_id, dataset_name, embeddings)
    # create retriever from db
    retriever = deeplake_db.as_retriever()
    return deeplake_db, retriever
