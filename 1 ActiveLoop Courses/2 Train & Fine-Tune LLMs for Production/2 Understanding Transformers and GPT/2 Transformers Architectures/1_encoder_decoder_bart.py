from transformers import AutoModel, AutoTokenizer

BART = AutoModel.from_pretrained("facebook/bart-large", cache_dir="cache")
print(BART)

