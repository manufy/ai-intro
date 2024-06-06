from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

text = "What would be a good company name for a company that makes colorful socks?"

print(llm(text))