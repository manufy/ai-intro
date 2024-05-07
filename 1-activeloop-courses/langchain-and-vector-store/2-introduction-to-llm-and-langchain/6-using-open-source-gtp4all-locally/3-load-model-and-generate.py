# pip install gpt4all
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

model = "./models/ggml-model-q4_0.bin"
#model = "./llama.cpp/models/gpt4all-lora-quantized.bin"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=model, callback_manager=callback_manager, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)