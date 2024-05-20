import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import requests

# imprime un texto en el centro de una imagen en negro por defecto

def print_text_on_image_centered(image, text, color="black"):
    # Crea un objeto Draw para la imagen
    draw = ImageDraw.Draw(image)
    
    
     # Define el tamaño inicial de la fuente
    font_size = 30
    font = ImageFont.load_default().font_variant(size=font_size)
    
    # Calcula las dimensiones del texto
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Reduce el tamaño de la fuente hasta que el texto se ajuste dentro de la imagen
    while text_width > image.width:
        font_size -= 1
        font = ImageFont.load_default().font_variant(size=font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
    
    # Calcula la posición del texto
    text_x = (image.width - text_width) / 2
    text_y = (image.height - text_height) / 2
    
    # Dibuja el texto en la imagen
    draw.text((text_x, text_y), text, font=font, fill=color)
    return image

# Crea una imagen en blanco por defecto

def create_background_image(width, height, color="white"):
    return Image.new("RGB", (width, height), color)
    



# Valida el token de la API de Hugging Face contra su API whoami-v2

def hf_validate_api_token(api_token):
    
    # Define la URL de la API
    url = "https://huggingface.co/api/whoami-v2"

    # Define los encabezados de la solicitud
    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    # Realiza la solicitud a la API
    response = requests.get(url, headers=headers)
    print(response.content)

    # Si la respuesta tiene un código de estado 200, el token es válido
    if response.status_code == 200:
        return True, response.json()['fullname']
    else:
        return False, response.json()['error']

def segment_gradio_image(api_token, model, image):
    print("api_token: " + api_token)
    is_token_valid, result = hf_validate_api_token(api_token)
    print("is_token_valid: " + str(is_token_valid))
    print("result: " + str(result))
    if is_token_valid == False:
        text_image = print_text_on_image_centered(create_background_image(500, 500, "white"), 'HuggingFace API Token invalid. Please enter a valid token.', 'red')
    else:
        text_image = print_text_on_image_centered(create_background_image(500, 500, "white"), 'PROCESANDO', 'blue')
  
    #response = segment_image(image)
    # Crea una imagen en blanco de 500x500 píxeles
    
    
    
    
    return text_image

# Create the Gradio interface
interface = gr.Interface(
    fn=segment_gradio_image, 
    inputs=[
        gr.Textbox(
            label="API Token", 
            placeholder="Enter your Hugging Face API token here"
            
        ), 
        gr.Textbox(
            label="AI Segmentation model", 
            placeholder="Enter your Segmentation model here",
            value="facebook/mask2former-swin-tiny-coco-panoptic"
            
        ), 
        "image"
    ],
    outputs="image",
    live=True,
    title="Segment Image",
    description="Upload an image and let the model segment it.",
    allow_flagging=False,
    examples=[
        ["", "https://api-inference.huggingface.co/models/facebook/mask2former-swin-tiny-coco-panoptic"]
    ]
)

# Launch the interface
interface.launch()