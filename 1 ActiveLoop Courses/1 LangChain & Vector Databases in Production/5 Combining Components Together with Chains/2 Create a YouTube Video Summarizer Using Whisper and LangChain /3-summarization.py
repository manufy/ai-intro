#from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
#from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)



# creates an instance of the RecursiveCharacterTextSplitter
# class, which is responsible for splitting input text into smaller chunks. 

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

# It is configured with a chunk_size of 1000 characters, 
# no chunk_overlap, and uses spaces, commas, and newline characters as separators.
# This ensures that the input text is broken down into manageable pieces, 
# allowing for efficient processing by the language model.

from langchain.docstore.document import Document

with open('text.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]

# Each Document object is initialized with the content of a chunk from the texts list.
# The [:4] slice notation indicates that only the first four chunks will be used 
# to create the Document objects. 

from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
from termcolor import colored
print ("----- SUMMARY -----")
print(colored(wrapped_text, 'yellow'))

# With the following line of code, we can see the prompt template
# that is used with the map_reduce technique.
# Now we’re changing the prompt and using another summarization method

print ("------ PROMPT TEMPLATE ------")


print(colored(chain.llm_chain.prompt.template, 'yellow'))

# The "stuff" approach is the simplest and most naive one, 
# in which all the text from the transcribed video is used in a single prompt. 
# This method may raise exceptions if all text is longer than the available 
# context size of the LLM and may not be the most efficient way to handle large amounts of text. 
# We’re going to experiment with the prompt below.
# This prompt will output the summary as bullet points.

prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, 
                        input_variables=["text"])

#  Also, we initialized the summarization chain using the stuff as chain_type and the prompt above.

chain = load_summarize_chain(llm, 
                             chain_type="stuff", 
                             prompt=BULLET_POINT_PROMPT)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary, 
                             width=1000,
                             break_long_words=False,
                             replace_whitespace=False)
print ("----- CONCISE SUMMARY -----")
print(colored(wrapped_text, 'yellow'))

# The 'refine' summarization chain is a method for generating more accurate
# and context-aware summaries. This chain type is designed to iteratively
# refine the summary by providing additional context when needed.
# That means: it generates the summary of the first chunk. 
# Then, for each successive chunk, the work-in-progress 
# summary is integrated with new info from the new chunk.

chain = load_summarize_chain(llm, chain_type="refine")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print("----- REFINED SUMMARY -----")
print(colored(wrapped_text, 'yellow'))