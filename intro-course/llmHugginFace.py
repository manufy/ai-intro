import langchain
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name_or_path = "meta-llama/Meta-Llama-3-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

llm = LangChain(model=model, tokenizer=tokenizer)

prompt = "Your prompt goes here."
generated_text = llm.generate(prompt)
print(generated_text)