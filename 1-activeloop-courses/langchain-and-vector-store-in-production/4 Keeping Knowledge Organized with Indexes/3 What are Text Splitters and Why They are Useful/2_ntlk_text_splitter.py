# The NLTKTextSplitter in LangChain is an implementation 
# of a text splitter that uses the Natural Language Toolkit (NLTK) library 
# to split text based on tokenizers. The goal is to split long texts
# into smaller chunks without breaking the structure of sentences and paragraphs.

from langchain.text_splitter import NLTKTextSplitter

# Load a long document
with open('/home/cloudsuperadmin/scrape-chain/langchain/LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

text_splitter = NLTKTextSplitter(chunk_size=500)
texts = text_splitter.split_text(sample_text)
print(texts)