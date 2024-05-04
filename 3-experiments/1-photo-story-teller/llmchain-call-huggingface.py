# import
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv(find_dotenv())

# Initialize the Hugging Face model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Initialize LLMChain with the Hugging Face model and tokenizer
llm = LLMChain(model=model, tokenizer=tokenizer)

# Let LLMChain generate for one input
prompt = "Tell me a joke"
generated_text = llm.generate(prompt)
print(generated_text)

# Let LLMChain generate for multiple inputs
inputs = ["Tell me a joke", "Write me a song"]
generated_results = llm.generate(inputs)
print(generated_results)
