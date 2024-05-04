from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import PromptTemplate, LLMChain

model_name = "microsoft/Phi-3-mini-4k-instruct"  # reemplace esto con el nombre del modelo que desea usar, por ejemplo, "gpt2", "gpt2-medium", "gpt2-large", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=model, prompt=prompt)