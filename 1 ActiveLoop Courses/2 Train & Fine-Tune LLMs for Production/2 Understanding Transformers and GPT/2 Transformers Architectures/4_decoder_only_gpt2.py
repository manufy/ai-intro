from transformers import AutoModel, pipeline

gpt2 = AutoModel.from_pretrained("gpt2", cache_dir="cache")
print(gpt2)


generator = pipeline(model="gpt2")
output = generator("This movie was a very", do_sample=True, top_p=0.95, num_return_sequences=4, max_new_tokens=50, return_full_text=False)

for item in output:
  print(">", item['generated_text'])