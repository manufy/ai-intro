
# The Recursive Character Text Splitter is a text splitter designed 
# to split the text into chunks based on a list of characters provided. 
# It attempts to split text using the characters from a list in order
# until the resulting chunks are small enough. By default, the list
# of characters used for splitting is ["\n\n", "\n", " ", "]

# To use the RecursiveCharacterTextSplitter, you can create an instance 
# of it and provide the following parameters:
# chunk_size : The maximum size of the chunks,
# as measured by the length_function (default is 100).
# chunk_overlap: The maximum overlap between chunks to maintain continuity
# between them (default is 20).
# length_function: parameter is used to calculate the length of the chunks.
# By default, it is set to len, which counts the number of characters in a chunk. However, you can also pass a token counter or any other function that calculates the length of a chunk based on your specific requirements.
# Using a token counter instead of the default len function
# can benefit specific scenarios, such as when working with language models
# with token limits. For example, OpenAI's GPT-3 has a token limit of 4096 tokens per request, 
# so you might want to count tokens instead of characters to better manage
# and optimize your requests.

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("The One Page Linux Manual.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
)

docs = text_splitter.split_documents(pages)
for doc in docs:
    print(doc)
    
# In this example, the text is loaded from a file,
# and the RecursiveCharacterTextSplitter is used to split it into chunks
# with a maximum size of 50 characters and an overlap of 10 characters. 
# The output will be a list of documents containing the split text.

# To use a token counter, you can create a custom function that calculates
# the number of tokens in a given text and pass it as the length_function parameter.
# This will ensure that your text splitter calculates the length of chunks based
# on the number of tokens instead of the number of characters. 