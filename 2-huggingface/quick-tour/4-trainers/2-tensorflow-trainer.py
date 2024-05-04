# To solve keras errors pip install tf-keras
# To disable warnings set TOKENIZERS_PARALLELISM=(true | false)

# You’ll start with a TFPreTrainedModel or a tf.keras.Model:

from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

# Load a preprocessing class like a tokenizer, image processor, feature extractor, or processor:

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Load a dataset:

from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes") 

# Create a function to tokenize the dataset:

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])  

# Apply the tokenizer over the entire dataset with map and then pass the dataset and tokenizer to prepare_tf_dataset(). You can also change the batch size and shuffle the dataset here if you’d like:

dataset = dataset.map(tokenize_dataset)  
tf_dataset = model.prepare_tf_dataset(
    dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
) 


# When you’re ready, you can call compile and fit to start training. Note that Transformers models all have a default task-relevant loss function, so you don’t need to specify one unless you want to:

from tensorflow.keras.optimizers import Adam 

model.compile(optimizer='adam')  # No loss argument!
model.fit(tf_dataset) 