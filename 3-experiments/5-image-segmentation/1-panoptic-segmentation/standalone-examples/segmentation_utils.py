import requests
from pycocotools import mask
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps, ImageFont
from dotenv import find_dotenv, load_dotenv
import os
import base64
import io
import random
import numpy as np
import cv2
from image_utils import print_text_on_image_centered, create_background_image
from icecream import ic


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/facebook/mask2former-swin-tiny-coco-panoptic"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

def segment_image_from_path(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def segment_image_from_image(image):
    # Convert the image to bytes
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    data = im_buf_arr.tobytes()

    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def decode_mask(mask_str, size):
    mask_data = base64.b64decode(mask_str)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_image = mask_image.resize(size).convert("L")
    return mask_image

def overlay_masks_on_image(image, segments, transparency=0.4):
    # Convert numpy array to PIL Image
    original_image = Image.fromarray(image).convert("RGBA")
 
    #original_image = Image.open(image).convert("RGBA")
    overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
    
    # Nueva capa para el texto

    text_layer = Image.new("RGBA", original_image.size, (255, 255, 255, 0))  
    
    for segment in segments:
        
        
        print(segment['label'] + " " + str(segment['score']))
        mask_str = segment['mask']
        mask_image = decode_mask(mask_str, original_image.size)
        color = generate_random_color()
        
        color_mask = ImageOps.colorize(mask_image, black="black", white=color)
        color_mask.putalpha(mask_image)
        
        overlay = Image.alpha_composite(overlay, color_mask)
        
        # Calcula el centroide de la mascara
        
        x, y = np.where(np.array(mask_image) > 0)
        centroid_x = x.mean()
        centroid_y = y.mean()
        
        # Imprime la etiqueta y la puntuación en la capa de texto
        
        font_size = 30
        draw = ImageDraw.Draw(text_layer)
        font_path = "/System/Library/Fonts/Arial.ttf"  # Path to Arial font on macOS
        font = ImageFont.truetype(font_path, font_size)
        label = segment['label']
        score = segment['score']
        text =f"{label}: {score}"
        
        # Estima el tamaño del texto hard rockandroll way
       
        text_width = 500
        text_height = 100
        draw.text((centroid_x - text_width / 2, centroid_y - text_height / 2), text, fill=(255, 255, 255, 255), font=font)
        
    
    # Ajusta la transparencia de la capa de superposición
    
    overlay = Image.blend(original_image, overlay, transparency)

    # Combina la capa de superposición con la capa de texto
    
    final_image = Image.alpha_composite(overlay, text_layer)
    
    #final_image = print_text_on_image_centered(final_image, 'SEGMENTING OK', 'green')
    
    return final_image

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def segment_and_overlay_results(image, api_token, model):
    #segments = segment_image_from_image(image)
    #final_image = overlay_masks_on_image(image, segments)
    #return final_image
    processed_image = None  # Initialize processed_image
    segments = []
    try:
        #segments = segment_image_from_image(image)
        #processed_image = overlay_masks_on_image(image, segments) 
        
        # debug image contents
        
        ic(image)
          
        if image.startswith('http://') or image.startswith('https://'):
            ic("image is a URL: " + image)
            response = requests.get(image)
            image = Image.open(BytesIO(response.content))
        else:
            # Check if image is a local file
           
          
            if os.path.isfile(os.path.join(os.getcwd(), image)):
                ic("image is a file: " + image + "OK")
                image = Image.open(image)
            else:
                raise ValueError("The image is neither a URL nor a local file.")
               
        
        
        #if os.path.isfile(image):
       #     ic ("--- image is a file ---")
       #     image = Image.open(image)
       # if image is None:
       #     ic("image is None")
       #     return None, []
        print(image)
        ic("--- calling segment_image_from_image ---")    
        #segments = segment_image_from_image(image)
        segments = segment_image_from_path('cats.jpg')
        for segment in segments:
            print("segmentation_utils.py segment_and_overlay_results")
            print(segment['label'] + " " + str(segment['score']))
        processed_image = print_text_on_image_centered(
                    create_background_image(500, 500, "white"),
                    'SEGMENTING OK',
                    'green'
                )
        print("--- calling overlay_masks_on_image ---")
        processed_image = overlay_masks_on_image(image, segments)
    except Exception as e:
        ic(e)
        processed_image = print_text_on_image_centered(
                    create_background_image(500, 500, "white"),
                    e,
                    'green'
                )
        segments = []
    finally:
        return processed_image, segments