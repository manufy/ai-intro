from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import LLMChain


# Initialize Hugging Face model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="./custom_cache")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir="./custom_cache")



# Example: Generate text using LLMChain
prompt = "Tell me a joke"
# Initialize LLMChain with the Hugging Face model and tokenizer
#llm = LLMChain(prompt=prompt, tokenizer=tokenizer)
#generated_text = llm.generate(prompt)
#print("Generated Text:", generated_text)

