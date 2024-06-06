
from fetch_article_function import get_article


from langchain.schema import (
    HumanMessage
)

article = get_article()
article_title = article.title
article_text = article.text

# we include several examples that guide the model's summarization process to generate bullet lists.
# As a result, the model is expected to generate a bulleted list summarizing the given article.

# prepare template for prompt
template = """
As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
- Climate change is causing a rise in global temperatures.
- This leads to melting ice caps and rising sea levels.
- Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
- Artificial Intelligence (AI) has developed significantly over the past decade.
- AI is now used in multiple fields such as healthcare, finance, and transportation.
- The future of AI is promising but requires careful regulation.

Now, here's the article you need to summarize:

==================
Title: {article_title}

{article_text}
==================

Please provide a summarized version of the article in a bulleted list format.
"""

# Format the Prompt
prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]

# The next step is to use ChatOpenAI class to load the GPT-4 model for generating the summary. 
# Then, the formatted prompt is passed to the language model as the input/prompt. 
# The ChatOpenAI class's chat instance takes a HumanMessage list as an input argument.

#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# generate summary
summary = chat(messages)
print(summary.content)