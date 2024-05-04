print("----- setting up application -----")
from dotenv import find_dotenv, load_dotenv
import os

print("----- setting up log level to error and ignoring user warnings -----")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

print ("----- setting up environment variables -----")
load_dotenv(find_dotenv())


print ("----- setting up langchain -----")
from langchain import LLMChain, OpenAI
from langchain import PromptTemplate


print("----- setting up transformers framework -----")

from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# img2text

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    return text

# llm

def generate_story(scenario):
    template = """
    You are a srory teller;
    You can generate a short story based on a simple narrative,
    the story should be no more than 20 words; 

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    #story_llm = LLMChain(llm="gpt2", max_length=20, prompt=prompt, verbose=True)
    #gpt2_model = GPT2LanguageModel() 
    #story_llm = LLMChain(llm=gpt2_model, prompt=prompt, verbose=True)


    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    #story_llm = LLMChain(llm=model, tokenizer=tokenizer, prompt=prompt, verbose=True)
    story_llm = LLMChain(llm={"model": model})
    #story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    return story

# text to speech
print("----- calling img2text -----")
# it can use URL too: text = img2text("https://images.pexels.com/photos/10599131/pexels-photo-10599131.jpeg?auto=compress&cs=tinysrgb&w=1200&lazy=load")
#scenario = img2text("photo.jpg")
print("----- img2text scenario output -----")
scenario="a woman in a blue dress holding a basket"
print(scenario)

print ("----- llm story output -----")
story=generate_story(scenario)
print(story)