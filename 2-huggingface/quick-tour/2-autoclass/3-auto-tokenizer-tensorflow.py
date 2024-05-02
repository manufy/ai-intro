from transformers import AutoTokenizer

# Load a tokenizer with AutoTokenizer:

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tensorflow tokenizer

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="tf",
)

# Pass your text to the tokenizer:

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")

# The tokenizer returns a dictionary containing:
# input_ids: numerical representations of your tokens.
# attention_mask: indicates which tokens should be attended to.

print(encoding)