from transformers import AutoModel, pipeline

BERT = AutoModel.from_pretrained("bert-base-uncased", cache_dir="cache")
print(BERT)

classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
lbl = classifier("""This restaurant is awesome.""")

print(lbl)