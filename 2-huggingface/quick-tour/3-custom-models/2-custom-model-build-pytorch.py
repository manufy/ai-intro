# Create a model from your custom configuration with AutoModel.from_config():

from transformers import AutoModel
from transformers import AutoConfig

my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)

my_model = AutoModel.from_config(my_config)