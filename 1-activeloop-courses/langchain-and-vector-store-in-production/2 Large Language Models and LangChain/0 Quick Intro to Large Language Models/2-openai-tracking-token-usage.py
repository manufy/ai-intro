from langchain_openai import OpenAI
from langchain.callbacks import get_openai_callback
from termcolor import colored

print(colored("----- Set up LLM and callback -----", 'green'))

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print ("---- Callback result ----")
    print(cb)

print(colored("----- Extracting cost from callback -----", 'green'))

total_cost_full_precision = format(float(str(cb).split("Total Cost (USD): ")[1].split("\n")[0].replace("$", "")), '.20f')
print("Total Cost Full Precision(USD): $" + total_cost_full_precision)
