# ai-intro

This repo contains small projects using current AI frameworks 

Updated to May 2024

This frameworks, tools and models are used:

- Langchain framework
- DeepLake vector datastore
- Local Ollama/Orca/Phi3 models
- HuggingFace LLM, sentiment and speech recognition models

Ensure to have about 50Gb free for HuggingFace and Ollama model downloads

## Environment install

- Create and activate conda environment (optional)

    `environment-and-requirements/create-conda-env.sh
    environment-and-requirements/activate-conda-env.sh`

- Install ActiveLoop dependencies
    
    `environment-and-requirements/install-activeloop-requirements.sh`

- Instal HuggingFace dependencies

    `environment-and-requirements/install-huggingface-requirements.sh`

- Create a .env file with API keys and credentials

    To be documented later using load_dotenv module

## Deprecation warnings

The most annoying thing is that the activeloop course is using deprecated modules.
The code will use the updated modules, but still some things needs to be changed to remove all deprecation warnings.
This depend on third party updates, despite the samples are using latest modules, if a module depends on another module, this can throw a warning because the included module was still not updated at the time of writing this repo
There are comments on the code to ignore the warnings to have a nice clean output.

## Courses

- [ActiveLoop LangChain & Vector Databases in Production](https://learn.activeloop.ai/courses/langchain)


## Platforms

- [HuggingFace Quick Tour](https://huggingface.co/docs/transformers/quicktour)

- [OpenAI Platform API](https://platform.openai.com)

- [ActiveLoop DataLake](https://app.activeloop.ai)

## LangChain tutorials

- [DataCamp](https://www.datacamp.com/tutorial/how-to-build-llm-applications-with-langchain)

- [Langchain Docs](https://python.langchain.com/)

## Models

- [OLlama](https://ollama.com/download)

