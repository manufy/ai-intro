from transformers import pipeline

# Start by creating an instance of pipeline() and specifying a task you want to use it for. 
# In this guide, you’ll use the pipeline() for sentiment analysis as an example:

classifier = pipeline("sentiment-analysis")

# The pipeline() downloads and caches a default pretrained model and tokenizer for sentiment analysis.
# Now you can use the classifier on your target text:

classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]

# If you have more than one input, pass your inputs as a list to the pipeline() to return a list of dictionaries:

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

