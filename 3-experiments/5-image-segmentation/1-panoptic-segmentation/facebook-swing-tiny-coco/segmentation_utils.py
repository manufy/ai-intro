import requests
from pycocotools import mask
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps, ImageFont
import os
import base64
import io
import random
import numpy as np
import cv2
from image_utils import print_text_on_image_centered, create_background_image
from icecream import ic
import traceback
from pprint import pprint




# Función para transformar la entrada en un array de numpy
# Si la entrada es una URL, descarga la imagen y la convierte en un array de numpy
# Si la entrada es una ruta de archivo, carga la imagen y la convierte en un array de numpy
# Si la entrada ya es un array de numpy, devuélvela tal cual
# Si la entrada no es ninguna de las anteriores, lanza un ValueError

def transform_image_to_numpy_array(input):
    if isinstance(input, np.ndarray):
        # Si la entrada es un array de numpy, devuélvela tal cual
        h, w = input.shape[:2]
        new_height = int(h * (500 / w))
        return cv2.resize(input, (500, new_height))
    elif isinstance(input, str):
        # Si la entrada es una cadena, podría ser una URL o una ruta de archivo
        if input.startswith('http://') or input.startswith('https://'):
            # Si la entrada es una URL, descarga la imagen y conviértela en un array de numpy
            # se necesita un header para evitar el error 403
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
            response = requests.get(input, headers=headers)
            ic(response.status_code)
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, -1)
            
            # Si la imagen tiene 3 canales (es decir, es una imagen en color),
            # convertirla de BGR a RGB
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image).convert("RGBA") 
            image = np.array(image)
        else:
            # Si la entrada es una ruta de archivo, carga la imagen y conviértela en un array de numpy
            image = cv2.imread(input)
       
        h, w = image.shape[:2]
        new_height = int(h * (500 / w))
        return cv2.resize(image, (500, new_height))
    else:
        raise ValueError("La entrada no es un array de numpy, una URL ni una ruta de archivo.")

def transform_image_to_numpy_array2(input):
    if isinstance(input, np.ndarray):
        # Si la entrada es un array de numpy, devuélvela tal cual
        return cv2.resize(input, (500, 500))
    elif isinstance(input, str):
        # Si la entrada es una cadena, podría ser una URL o una ruta de archivo
        if input.startswith('http://') or input.startswith('https://'):
            # Si la entrada es una URL, descarga la imagen y conviértela en un array de numpy
            # se necesita un header para evitar el error 403
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
            response = requests.get(input, headers=headers)
            ic(response.status_code)
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, -1)
            
             # Si la imagen tiene 3 canales (es decir, es una imagen en color),
             # convertirla de BGR a RGB
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image).convert("RGBA") 
            image = np.array(image)
        else:
            # Si la entrada es una ruta de archivo, carga la imagen y conviértela en un array de numpy
            image = cv2.imread(input)
       
        return cv2.resize(image, (500, 500))
    else:
        raise ValueError("La entrada no es un array de numpy, una URL ni una ruta de archivo.")
    
def segment_image_from_numpy(image_array, api_token, model):
    
    #API_URL = "https://api-inference.huggingface.co/models/facebook/mask2former-swin-tiny-coco-panoptic"
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_token}"}
    ic(API_URL)
    ic(headers)
    # Convert the image to bytes
    is_success, im_buf_arr = cv2.imencode(".jpg", image_array)
    data = im_buf_arr.tobytes()
    response = requests.post(API_URL, headers=headers, data=data)
    pprint(response.json())
    return response.json()  


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
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    original_image = image
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    
    overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
    text_layer = Image.new("RGBA", original_image.size, (255, 255, 255, 0))  
    
    for segment in segments:
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
        font = ImageFont.load_default().font_variant(size=font_size)
        label = segment['label']
        score = segment['score']
        text =f"{label}: {score}"
        
        # Calcula el tamaño del texto
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Asegúrate de que las coordenadas del texto están dentro de los límites de la imagen
        text_x = max(0, min(centroid_x - text_width / 2, original_image.size[0] - text_width))
        text_y = max(0, min(centroid_y - text_height / 2, original_image.size[1] - text_height))

        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    # Ajusta la transparencia de la capa de superposición
    overlay = Image.blend(original_image, overlay, transparency)

    # Combina la capa de superposición con la capa de texto
    final_image = Image.alpha_composite(overlay, text_layer)
    
    return final_image








def overlay_masks_on_image2(image, segments, transparency=0.4):
    # Convert numpy array to PIL Image
    #original_image = Image.fromarray(image).convert("RGBA")
    #original_image = image
    #original_image = Image.open(image).convert("RGBA")
    # para file es str
    # para url es numpy.ndarray
    # para cv.imread es numpy.ndarray
    
    # Convertir el array de numpy a una imagen PIL si es necesario
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    print(type(image))
    print(image)
    original_image = image
    
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    
    print(original_image.size)
    overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
    print(overlay.size)
    # Nueva capa para el texto

    text_layer = Image.new("RGBA", original_image.size, (255, 255, 255, 0))  
    
    for segment in segments:
        
        
        print(segment['label'] + " " + str(segment['score']))
        mask_str = segment['mask']
        mask_image = decode_mask(mask_str, original_image.size)
        
        
        
        # Convierte la imagen de la máscara a un array de numpy
        mask_array = np.array(mask_image)

        # Encuentra los píxeles blancos
        y, x = np.where(mask_array > 0)

        # Calcula el cuadro delimitador de los píxeles blancos
        x_min, y_min, width, height = cv2.boundingRect(np.array(list(zip(x, y))))

    
        # Crea un objeto ImageDraw para dibujar en la imagen original
        draw = ImageDraw.Draw(original_image)   
        
                
        # Dibuja el cuadro delimitador en la imagen original
        draw.rectangle([(x_min, y_min), (x_min + width, y_min + height)], outline=(0, 255, 0), width=2)
                        
        
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
        
        
        # Asegúrate de que las coordenadas del texto están dentro de los límites de la imagen
        text_x = max(0, min(centroid_x - text_width / 2, original_image.size[0] - text_width))
        text_y = max(0, min(centroid_y - text_height / 2, original_image.size[1] - text_height))
# Asegúrate de que las coordenadas del texto están dentro de los límites de la imagen
        text_x = max(0, min(centroid_x, original_image.size[0] - text_width))
        text_y = max(0, min(centroid_y, original_image.size[1] - text_height))


        # Calcula las coordenadas del texto
        text_x = centroid_x - text_width / 2
        text_y = centroid_y - text_height / 2
        
        
        # Asegúrate de que las coordenadas del texto están dentro de los límites de la imagen
        text_x = max(0, min(text_x, original_image.size[0] - text_width))
        text_y = max(0, min(text_y, original_image.size[1] - text_height))
        
        
        draw.text((centroid_x - text_width / 2, centroid_y - text_height / 2), text, fill=(255, 255, 255, 255), font=font)

        #draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    # Ajusta la transparencia de la capa de superposición
    print(original_image.size)
    print(overlay.size)
    overlay = Image.blend(original_image, overlay, transparency)

    # Combina la capa de superposición con la capa de texto
    
    final_image = Image.alpha_composite(overlay, text_layer)
    
    #final_image = print_text_on_image_centered(final_image, 'SEGMENTING OK', 'green')
    
    return final_image

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def segment_and_overlay_results(image_path, api_token, model):
    #segments = segment_image_from_image(image)
    #final_image = overlay_masks_on_image(image, segments)
    #return final_image
    processed_image = None  # Initialize processed_image
    segments = []
    #image_type = None
    #if isinstance(image_path, str):
    #    image_type = 'FILE'
    #    image = cv2.imread('cats.jpg')
    #elif isinstance(image_path, np.ndarray):
    #    image_type = 'NUMPY ARRAY'
    #else:
    #    raise ValueError("The image is neither a Image nor a local file.")  
          
    #ic(image_type)
    image = transform_image_to_numpy_array(image_path)
    # imprime tres primeros pixeles
    print(type(image))
    ic(image[0, 0:3])
    
    
    
    
    try:
        #segments = segment_image_from_image(image)
        #processed_image = overlay_masks_on_image(image, segments) 
        
        # debug image contents

        #if os.path.isfile(image):
       #     ic ("--- image is a file ---")
       #     image = Image.open(image)
       # if image is None:
       #     ic("image is None")
       #     return None, []
        
        ic("--- calling segment_image_from_path ---")    
        segments = segment_image_from_numpy(image, api_token, model)
        #if image_type == 'FILE':
        #    segments = segment_image_from_path(image_path)
        #if image_type == 'NUMPY ARRAY':
        #    segments = segment_image_from_image(image_path)
            
        ic("--- printing segments ---")
        for segment in segments:    
            ic(segment['label'] ,segment['score'])
        processed_image = print_text_on_image_centered(
                    create_background_image(500, 500, "white"),
                    'SEGMENTING OK',
                    'green'
                )
        ic("--- calling overlay_masks_on_image ---")
        processed_image = overlay_masks_on_image(image, segments)
        return processed_image, segments
    except Exception as e:
        print("EXCEPTION")
        ic(e)
        print(traceback.format_exc())
        processed_image = print_text_on_image_centered(
                    create_background_image(500, 500, "white"),
                    e,
                    'green'
                )
        segments = []
        return processed_image, segments
    #finally:
    #return processed_image, segments