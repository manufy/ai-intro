from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch import nn

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Now pass your preprocessed batch of inputs directly to the model. You just have to unpack the dictionary by adding **:

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

pt_outputs = pt_model(**pt_batch)

# The model outputs the final activations in the logits attribute. Apply the softmax function to the logits to retrieve the probabilities:

pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)