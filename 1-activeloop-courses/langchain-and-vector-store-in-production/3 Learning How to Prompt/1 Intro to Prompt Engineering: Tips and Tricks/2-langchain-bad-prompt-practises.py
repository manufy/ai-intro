from langchain_core.prompts import PromptTemplate

template = "Tell me something about {topic}."
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)
prompt.format(topic="dogs")
print("----- Bad prompting, no context -----")
print(prompt.format(topic="dogs"))