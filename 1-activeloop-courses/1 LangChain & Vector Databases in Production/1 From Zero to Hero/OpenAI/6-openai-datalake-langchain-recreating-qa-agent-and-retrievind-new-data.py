from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = os.environ["ACTIVELOOP_DATASET_OPENAI"]
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# load the existing Deep Lake dataset and specify the embedding function
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# instantiate the wrapper class for GPT3
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# create a retriever from the db
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# create an agent that uses the tool
agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.run("When was Michael Jordan born?")
print(response)