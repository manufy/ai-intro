from langchain.chat_models import AzureChatOpenAI

BASE_URL = "<URL>"
API_KEY = db.secrets.get("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = "<deployment_name>"
llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    streaming=True,
    verbose=True,
    temperature=0,
    max_tokens=1500,
    top_p=0.95
)
