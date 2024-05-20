import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# imprime un texto en el centro de una imagen en negro por defecto

def print_text_on_image_centered(image, text, color="black"):
    # Crea un objeto Draw para la imagen
    draw = ImageDraw.Draw(image)
    # Define el texto y la fuente
    text = "HuggingFace API Token invalid. Please enter a valid token."
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
    
def segment_gradio_image(api_token, image):
    print("api_token: " + api_token)
    #response = segment_image(image)
    # Crea una imagen en blanco de 500x500 píxeles
    
    
    text_image = print_text_on_image_centered(create_background_image(500, 500, "white"), 'HuggingFace API Token invalid. Please enter a valid token.', 'red')
  
    
    return text_image


# Create the Gradio interface
interface = gr.Interface(
    fn=segment_gradio_image, 
    inputs=[
        gr.Textbox(
            label="API Token", 
            placeholder="Enter your Hugging Face API token here"
            
        ), 
        "image"
    ],
    outputs="image",
    live=True,
    title="Segment Image",
    description="Upload an image and let the model segment it."
)

# Launch the interface
interface.launch()