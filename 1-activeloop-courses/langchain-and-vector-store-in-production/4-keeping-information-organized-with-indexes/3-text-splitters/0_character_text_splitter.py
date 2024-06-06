
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("The One Page Linux Manual.pdf")
pages = loader.load_and_split()

# By loading the text file, we can ask more specific questions related to the subject, 
# which helps minimize the likelihood of LLM hallucinations and
# ensures more accurate, context-driven responses.

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(pages)

print(texts[0])

print (f"You have {len(texts)} documents")
print ("Preview:")
print (texts[0].page_content)


# No universal approach for chunking text will fit all scenarios 
# what's effective for one case might not be suitable for another. 
# Finding the best chunk size for your project means going through a few steps.
# First, clean up your data by getting rid of anything that's not needed, 
# like HTML tags from websites. Then, pick a few different chunk sizes to test. 
# The best size will depend on what kind of data you're working with and 
# the model you're using.  Finally, test out how well each size works
# by running some queries and comparing the results. You might need to try
# a few different sizes before finding the best one. This process might 
# take some time, but getting the best results from your project is worth it.