#from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

loader = TextLoader('my_file.txt')
documents = loader.load()
print(documents)
# Output
# [Document(page_content='<FILE_CONTENT>', metadata={'source': 'file_path.txt'})]