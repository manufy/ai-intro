
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification


pt_save_directory = "./pt_save_pretrained"

tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)
tf_model = TFAutoModelForSequenceClassification.from_pretrained(pt_save_directory, from_pt=True)