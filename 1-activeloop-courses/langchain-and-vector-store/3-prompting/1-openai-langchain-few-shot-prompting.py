from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""

example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="yellow")

# Create LLMChain
chain = few_shot_prompt | llm

# Run the LLMChain to get the AI-generated response
response = chain.invoke(formatted_prompt)

print("Color: yellow")
print("Emotion:", response)