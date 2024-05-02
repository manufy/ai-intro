from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Now pass your preprocessed batch of inputs directly to the model. You just have to unpack the dictionary by adding **:

tf_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="tf",
)



tf_outputs = tf_model(tf_batch)

# The model outputs the final activations in the logits attribute. Apply the softmax function to the logits to retrieve the probabilities:

# huggingface says this, but it should bt tf.nn tf_predictions = tf.nn.softmax(tf_outputs.logits, dim=-1)

tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
print(tf_predictions)