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

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/facebook/mask2former-swin-tiny-coco-panoptic"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

def segment_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def decode_mask(mask_str, size):
    mask_data = base64.b64decode(mask_str)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_image = mask_image.resize(size).convert("L")
    return mask_image

def overlay_masks_on_image(image_path, segments, transparency=0.4):
    original_image = Image.open(image_path).convert("RGBA")
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
    
    
    return final_image

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def main():
    image_path = "cats.jpg"
    response = segment_image(image_path)

    if isinstance(response, list):
        overlayed_image = overlay_masks_on_image(image_path, response)
        overlayed_image.show()
        overlayed_image.save("overlayed_image.png")
    else:
        print("Error in segmentation:", response)

__main__ = main()