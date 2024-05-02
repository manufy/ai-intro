from transformers import TFAutoModel
from transformers import AutoConfig


my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)

my_model = TFAutoModel.from_config(my_config)