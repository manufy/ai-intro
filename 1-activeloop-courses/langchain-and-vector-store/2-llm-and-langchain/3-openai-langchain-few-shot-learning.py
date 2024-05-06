from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

# Define your examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    },
    {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

# Create the example template
example_template = """
User: {query}
AI: {answer}
"""

# Create the example prompt
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# Define prefix and suffix
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""

suffix = """
User: {query}
AI: """

# Create the FewShotPromptTemplate
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Create the runnable_sequence
runnable_sequence = few_shot_prompt_template | llm

# Invoke the sequence with your input prompt
result = runnable_sequence.invoke("What's the meaning of life?")

# Print the result
from termcolor import colored

# Print the content in yellow
print(colored(result.content, "yellow"))
