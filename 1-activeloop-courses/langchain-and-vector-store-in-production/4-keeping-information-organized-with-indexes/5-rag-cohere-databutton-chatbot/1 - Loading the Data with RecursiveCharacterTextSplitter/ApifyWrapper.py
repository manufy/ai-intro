from langchain.document_loaders import ApifyDatasetLoader
from langchain.utilities import ApifyWrapper
from langchain.document_loaders.base import Document
import os

os.environ["APIFY_API_TOKEN"] = db.secrets.get("APIFY_API_TOKEN")

apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "ENTER\YOUR\URL\HERE"}]},
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"] if dataset_item["text"] else "No content available",
        metadata={
            "source": dataset_item["url"],
            "title": dataset_item["metadata"]["title"]
        }
    ),
)

docs = loader.load()