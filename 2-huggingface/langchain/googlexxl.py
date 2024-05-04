from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(model_kwargs={"temperature": 0.5, "max_length": 64},repo_id="google/flan-t5-xxl")
prompt = "In which country is Tokyo?"
completion = llm(prompt)
print(completion)