
# Quick intro to llm invoke call , as shown in the course
# For speed reasons I do not to use a local model, neither OpenAI to avoid running costs

# required pip install sentence-transformers, pip install deeplake

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
#from langchain.embeddings import HuggingFaceEmbeddings

#from deeplake import DeepLake


import sys
import os

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.utils.hub')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')


# TODO: find a better way to handle stdout and stderr message suppression

print ("------ Setting up LLM ------")

stdout_original = sys.stdout
stderr_original = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Initialize the HuggingFace endpoint to use with mistral model




llm_model = "openai-community/gpt2"
#llm_model = "google-bert/bert-base-uncased"
llm_model = "bert-base-uncased"
try:
    repo_id = llm_model
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5
    )
finally:
    sys.stdout.close()
    sys.stdout = stdout_original
    sys.stderr.close()
    sys.stderr = stderr_original
    print("------ LLM Initialized ------")

print("------ Loading Embeddings ------")

 
 # TRANSFORMERS_CACHE will be deprecated in the future
 # but for some reason still not applied, so I must force this env to force embedding download to specified cache dir

custom_cache_dir="cache"
os.environ["TRANSFORMERS_CACHE"] = custom_cache_dir 
os.environ["HF_HOME"] = custom_cache_dir 

#embeddings = HuggingFaceEmbeddings(model_name=model,
#                                       model_kwargs={'device': 'cpu'}, encode_kwargs={'device': 'cpu'})

embeddings_model = HuggingFaceEmbeddings(model_name=llm_model)



print("------ Creating documents ------")

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

print("------ Documents ------")
print(docs)


print("------ Setting up tokenizer ------")

from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained(llm_model)
tokenizer.pad_token = tokenizer.eos_token


print("------ Generating embeddings ------")

# List to store embeddings for each document
document_embeddings = []


# Añade un nuevo token de padding
tokenizer.add_special_tokens({'pad_token': '[PAD]'})





# Genera las incrustaciones para cada documento
for document in docs:
     # Tokeniza el documento
    tokens = tokenizer(document.page_content, padding=True, truncation=True, max_length=512, return_tensors='pt')
    print(tokens)
    # Genera las incrustaciones para el documento usando el método embed_documents
    embedding = embeddings_model.embed_documents(tokens)
    document_embeddings.append(embedding)



print("------ Generated embeddings ------")






# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)

print("------ Creating DeepLake dataset ------")

my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = os.environ["ACTIVELOOP_DATASET"]
# Codifica cada texto individualmente
# Codifica cada texto individualmente
# Codifica cada documento individualmente
# Define el token de padding


# embeddings_tensors = embeddings.embed_documents([doc.page_content for doc in docs])

# Genera las incrustaciones

# Verifica si las incrustaciones se generaron correctamente
# Genera las incrustaciones
document_embeddings = embeddings_model.embed_documents([doc.page_content for doc in docs])
document_embeddings_tensor = torch.tensor(document_embeddings)

if not document_embeddings:
    raise ValueError("No se generaron incrustaciones.")

print("------ Embeddings ------")
print(document_embeddings)


dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

#db = DeepLake(dataset_path=dataset_path)
db = DeepLake(dataset_path=dataset_path, embedding=embeddings_model)

# add documents to our Deep Lake dataset
#db.add_documents(docs)