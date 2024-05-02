# You’ll start with a PreTrainedModel or a torch.nn.Module: 

from transformers import AutoModelForSequenceClassification    

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

# TrainingArguments contains the model hyperparameters you can change like learning rate, batch size, and the number of epochs to train for. The default values are used if you don’t specify any training arguments: 

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="path/to/save/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

# Load a preprocessing class like a tokenizer, image processor, feature extractor, or processor:

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Load a dataset:

from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes") 

# Create a function to tokenize the dataset:

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

# Then apply it over the entire dataset with map:

dataset = dataset.map(tokenize_dataset, batched=True)

# A DataCollatorWithPadding to create a batch of examples from your dataset:

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Now gather all these classes in Trainer:

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

trainer.train()