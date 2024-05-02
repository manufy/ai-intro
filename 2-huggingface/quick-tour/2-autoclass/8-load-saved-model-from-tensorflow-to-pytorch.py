
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tf_save_directory = "./tf_save_pretrained"

tokenizer = AutoTokenizer.from_pretrained(tf_save_directory)
pt_model = AutoModelForSequenceClassification.from_pretrained(tf_save_directory, from_tf=True)