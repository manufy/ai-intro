from PIL import Image, ImageDraw, ImageFont


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
    
