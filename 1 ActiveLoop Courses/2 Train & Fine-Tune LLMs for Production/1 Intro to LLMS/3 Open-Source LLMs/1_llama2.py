from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# download model
#model_id = "meta-llama/Llama-2-7b-chat-hf"
#model_id = "tiiuae/falcon-7b-instruct"
model_id = "databricks/dolly-v2-3b"

#tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="cache")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
	model_id,
	trust_remote_code=True,
	torch_dtype=torch.bfloat16
)


# generate answer
prompt = "Translate English to French: Configuration files are easy to use!"
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
outputs = model.generate(**inputs, max_new_tokens=100)

# print answer
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
