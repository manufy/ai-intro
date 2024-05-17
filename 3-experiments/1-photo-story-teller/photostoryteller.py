print("----- setting up application -----")
from dotenv import find_dotenv, load_dotenv
import os
import requests
import streamlit as st

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




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




def main():

    try:
        st.set_page_config(page_title="ManuAI Experiment 1 - Photo Story Teller", page_icon="📸", layout="wide")
        st.header("ManuAI Experiment 1 - Photo Story Generation and Audio Teller")

        st.write("Upload a photo and the application will generate a story based on the photo.")
        st.write("The application uses the Hugging Face API to generate a scenario based on the photo, and then uses the OpenAI API to generate a story based on the scenario.")
        st.write("The story is then converted usint a TTS model using the Hugging Face inference API to generate the audio.")

        st.write("Still WIP, this deploy is for demo purposes only.")
    
    
     


        uploaded_file = st.file_uploader("Choose a jpg base64 encoded photo...", type="jpg")
        if uploaded_file is not None:
            print("uploaded file")
            bytes_data = uploaded_file.getvalue()
            with open("photo.jpg", "wb") as f:
                f.write(bytes_data) 
            st.image(uploaded_file, caption="Uploaded photo", use_column_width=True)

            with st.spinner(text="Generating IMG2TEXT scenario..."):
                scenario = img2text(uploaded_file.name)
            st.success('Scenario generated!')   
            

            with st.spinner(text="Generating LLM story..."):
                story = generate_story(scenario)
            st.success('Story generated!')   

            with st.spinner(text="Generating Audio TTS output..."):
                text2speech(story)
            st.success('Audio generated!')   
        
            with st.expander("scenario"):
                st.write(scenario)
            with st.expander("story"):
                st.write(story)  
            st.audio("audio.flac")
    except:
         st.error('Some unexpected exception has occurred, remember', icon="🚨")
        
if __name__ == "__main__":
    main()  
