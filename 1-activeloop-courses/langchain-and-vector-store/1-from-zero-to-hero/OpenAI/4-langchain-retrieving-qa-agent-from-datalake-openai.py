from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings
import os



llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = os.environ["ACTIVELOOP_DATASET_OPENAI"]
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)


retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
	retriever=db.as_retriever()
)

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.run("When was Napoleone born?")
print(response)