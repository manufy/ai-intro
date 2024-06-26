print("----- setting up application -----")
from dotenv import find_dotenv, load_dotenv
import os
import requests
import streamlit as st
from PIL import Image

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

def generate_story2(scenario):
    template = """
    You are a srory teller;
    You should generate a short story based on  {scenario};
    the story should be no more than 20 words;
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    
    print(prompt)

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


def generate_story(scenario):
    # Define the prompt template
    template = """
    You are a storyteller. Write a short story based on the following scenario:
    Scenario: {scenario}
    The story should be no more than 20 words.
    """
    # Format the prompt with the scenario
    prompt = template.format(scenario=scenario)
    
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set up text generation pipeline
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    # Generate the story
    
    prompt = f"Scenario: {scenario}\nWrite a short story in less than 20 words."

    
    story = generator(
        prompt,
        max_new_tokens=20,  # Limit the new tokens generated to 20 words
        truncation=True,
        num_return_sequences=1,
        return_full_text=False,
        temperature=0.7,  # Adjust the creativity of the output
        top_p=0.9,  # Nucleus sampling
        top_k=50  # Top-k sampling
    )[0]['generated_text']
    
    # Remove the prompt part from the output if necessary
    story = story.replace(prompt, "").strip()

    # Return the generated story, ensuring it is stripped of leading/trailing whitespace
    return story.strip()



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
        st.set_page_config(page_title="ManuAI Experiment 1 - Photo Story Teller :sunglasses:", page_icon="📸", layout="wide")
        st.markdown("*ManuAI* :green[Experiment 1] - :rainbow[Photo Story Generation and Audio Teller] :sunglasses:")

        st.markdown(":tulip: This *demo*  uses the Hugging Face API to generate a scenario / description based on the photo, and then uses the OpenAI API to generate a story based on the scenario.")
        st.markdown(":cherry_blossom: The story is then converted using a TTS model from the Hugging Face inference API to generate the audio.")

        st.markdown(":rose: :red[**Still WIP, this deploy is for demo purposes only.**]")

      
      
    
    
     


        uploaded_file = st.file_uploader(":hibiscus: Choose a photo...", type="jpg")
        if uploaded_file is not None:
            print("uploaded file")
            image = Image.open(uploaded_file)

            st.image(uploaded_file, caption="Uploaded photo", use_column_width=True)

            with st.spinner(text="Generating IMG2TEXT scenario... " + uploaded_file.name ):
                scenario = img2text(image)
            st.success(':sunflower: Scenario generated!')   
            

            with st.spinner(text="Generating LLM story..."):
                story = generate_story(scenario)
            st.success(':blossom: Story generated!')   

            with st.spinner(text="Generating Audio TTS output..."):
                text2speech(story)
            st.success(':tulip: Audio generated!')   
        
            with st.expander(":cherry_blossom: View scenario ... "):
                st.write(scenario)
            with st.expander(":rose: View story ..."):
                st.write(story)  
            st.audio("audio.flac")
    except Exception as e:
         st.error('Some unexpected exception has occurred, remember this is WIP, go for a beer and try lateer please :)', icon="🚨")
         st.error(e)
if __name__ == "__main__":
    main()  
