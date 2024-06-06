from langchain.text_splitter import RecursiveCharacterTextSplitter

# we split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)
docs_split = text_splitter.split_documents(docs)
