import requests
from pycocotools import mask
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from pprint import pprint
from dotenv import find_dotenv, load_dotenv
import os
import base64
import io
from pprint import pprint
#load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/facebook/mask2former-swin-tiny-coco-panoptic"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
#print(headers)

#def query(filename):
#    with open(filename, "rb") as f:
#        data = f.read()
#    response = requests.post(API_URL, headers=headers, data=data)
#    return response.json()

#s¡output = query("cats.jpg")
# Imprime el tipo de la variable 'output'
# ∫print(type(output))


def segment_image(image_path):
   with open(image_path, "rb") as f:
       data = f.read()
   response = requests.post(API_URL, headers=headers, data=data)
   return response.json()




def draw_segmented_image(image_path, segments):
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for segment in segments:
        mask = Image.open(io.BytesIO(segment['mask']['data']))
        mask = mask.resize(image.size)
        color = tuple(segment['color'])
        mask = mask.convert("L").point(lambda p: p > 128 and 255)
        overlay = Image.composite(Image.new("RGBA", image.size, color + (128,)), overlay, mask)

    combined = Image.alpha_composite(image, overlay)
    return combined

def decode_mask(mask_str):
    mask_data = base64.b64decode(mask_str)
    mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
    return mask_image

def overlay_mask_on_image(original_image, mask_image, color=(255, 0, 0), alpha=0.5):
    # Create a color version of the mask
    color_mask = ImageOps.colorize(mask_image, black="black", white=color)
    # Convert the mask to have an alpha channel
    color_mask.putalpha(mask_image)
    # Resize the mask to match the original image size
    color_mask = color_mask.resize(original_image.size, resample=Image.BILINEAR)
    # Composite the mask with the original image
    overlay_image = Image.alpha_composite(original_image.convert("RGBA"), color_mask)
    return overlay_image






def decode_mask(mask_str, size):
    mask_data = base64.b64decode(mask_str)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_image = mask_image.resize(size).convert("L")
    return mask_image

def decode_and_display_mask(mask_str):
    mask_data = base64.b64decode(mask_str)
    mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
    mask_image.show()  # Display the mask image


image_path = "cats.jpg"
response = segment_image(image_path)

if isinstance(response, list):  # Check if the result is a list of segments
        original_image = Image.open(image_path).convert("RGBA")
        for segment in response:
            label = segment['label']
            score = segment['score']
            mask = segment['mask']
            print(f"Label: {label}, Score: {score}")
            # decode_and_display_mask(mask)
            
            mask_image = decode_mask(mask, original_image.size)
            overlay_image = overlay_mask_on_image(original_image, mask_image)
            
            overlay_image.show()  # Display the image with overlay
            overlay_image.save(f"overlay_{label}.png")
else:
        print("Error in segmentation:", response)
#if 'segments' in segments:
#        segmented_image = draw_segmented_image(image_path, segments['segments'])
#        segmented_image.show()
#else:
#        print("Error in segmentation:", segments)