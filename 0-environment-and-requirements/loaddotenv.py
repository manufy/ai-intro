from dotenv import find_dotenv, load_dotenv
import os
#load_dotenv(find_dotenv())
print("Environment Variable HUGGINGFACEHUB_API_TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
