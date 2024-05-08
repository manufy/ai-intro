

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders."
    }, {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Create the LLMChain for the few_shot_prompt_template
# chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

chain = few_shot_prompt_template | llm

# Run the LLMChain with input_data
input_data = {"query": "How can I learn quantum computing?"}
response = chain.invoke(input_data)

print(response.content)