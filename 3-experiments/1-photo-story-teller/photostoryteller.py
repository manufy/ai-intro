print("----- setting up application -----")
from dotenv import find_dotenv, load_dotenv
import os

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline


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
    #story_llm = LLMChain(llm={"model": model})
    #story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)

    story_llm = prompt | model
    input_data = {"scenario": scenario}

    # Convierte el objeto StringPromptValue en un tensor de PyTorch
    #input_ids = tokenizer.encode(scenario, return_tensors='pt')

    # Set up text generation pipeline
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    story = generator(scenario, max_length=100, truncation=True, num_return_sequences=1)[0]['generated_text']

    #story = story_llm.invoke(input_data)
    return story.strip()

# text to speech
#print("----- calling img2text -----")
# it can use URL too: text = img2text("https://images.pexels.com/photos/10599131/pexels-photo-10599131.jpeg?auto=compress&cs=tinysrgb&w=1200&lazy=load")
#scenario = img2text("photo.jpg")
#print("----- img2text scenario output -----")
##scenario="a woman in a blue dress holding a basket"
#print(scenario)

#print ("----- llm story output -----")
#story=generate_story(scenario)
##print(story)

import requests

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {
	    "inputs": message
    }
    print("----- calling text2speech -----")
    response = requests.post(API_URL, headers=headers, json=payload)

   
    with open("audio.flac", "wb") as f:
        f.write(response.content)

#print("----- calling text2speech -----")

#text2speech("hello world")
import streamlit as st



def main():
    st.set_page_config(page_title="Photo Story Teller", page_icon="📸", layout="wide")
    st.header("Photo Story Teller")
    st.write(f"El token es: {HUGGINGFACEHUB_API_TOKEN}")
    uploaded_file = st.file_uploader("Choose a photo...", type="jpg")
    if uploaded_file is not None:
        print("uploaded file")
        bytes_data = uploaded_file.getvalue()
        with open("photo.jpg", "wb") as f:
            f.write(bytes_data) 
        st.image(uploaded_file, caption="Uploaded photo", use_column_width=True)
        st.write("Generating IMG2TEXT scenario...")
        scenario = img2text(uploaded_file.name)
        st.write("Generating LLM scenario...")
        story = generate_story(scenario)
        st.write("Audio TTS output...")
        text2speech(story)
        with st.expander("scenario"):
           st.write(scenario)
        with st.expander("story"):
            st.write(story)  
        st.audio("audio.flac")

if __name__ == "__main__":
    main()  
