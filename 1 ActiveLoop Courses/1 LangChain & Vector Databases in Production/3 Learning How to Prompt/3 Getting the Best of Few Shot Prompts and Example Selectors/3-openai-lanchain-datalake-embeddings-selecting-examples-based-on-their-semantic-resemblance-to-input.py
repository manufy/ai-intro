from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import DeepLake
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Create a PromptTemplate
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Define some examples
examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

# create Deep Lake dataset
# TODO: use your organization id here.  (by default, org id is your username)
import os 
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = os.environ["ACTIVELOOP_DATASET_FEWSHOTSAMPLE"]
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path)

# Embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Instantiate SemanticSimilarityExampleSelector using the examples
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embeddings, db, k=1
)

# Create a FewShotPromptTemplate using the example_selector
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the temperature from Celsius to Fahrenheit",
    suffix="Input: {temperature}\nOutput:", 
    input_variables=["temperature"],
)

# Test the similar_prompt with different inputs
print(similar_prompt.format(temperature="10°C"))   # Test with an input
print(similar_prompt.format(temperature="30°C"))  # Test with another input

# Add a new example to the SemanticSimilarityExampleSelector
similar_prompt.example_selector.add_example({"input": "50°C", "output": "122°F"})
print(similar_prompt.format(temperature="40°C")) # Test with a new input after adding the example