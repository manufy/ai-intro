import gradio as gr
import requests
from image_utils import print_text_on_image_centered, create_background_image
from hf_utils import hf_validate_api_token
from segmentation_utils import segment_and_overlay_results

def segment_gradio_image(api_token, model, image):
    
    # Validacion del token y la imagen
    
    is_token_valid, api_token_message = hf_validate_api_token(api_token)
    if not is_token_valid:
        text_image = print_text_on_image_centered(
            create_background_image(500, 500, "white"),
            'HuggingFace API Token invalid. Please enter a valid token.',
            'red'
        )
        segments_list = "No segments available."
    else:
        if image is None:
            text_image = print_text_on_image_centered(
                create_background_image(500, 500, "white"),
                'No image detected',
                'orange'
            )
            segments_list = "No segments available."
        else:
            text_image = print_text_on_image_centered(
                create_background_image(500, 500, "white"),
                'PROCESANDO',
                'blue'
            )
            segments_list = "No segments available."
            # Assuming segment_image is a placeholder for actual segmentation function
            # Uncomment and modify this part according to your segmentation implementation
            # response = segment_image(api_token, model, image)
            # text_image = response["segmented_image"]
            
            text_image, segments = segment_and_overlay_results(image,api_token, model)
            print("app.py segment_gradio_image")
            segments_list = "Segments:\n"
            for segment in segments:
                print(segment['label'] + " " + str(segment['score']))
                segments_list += f"\n{segment['label']}: {segment['score']}"
           
    
    return api_token_message, text_image, segments_list



with gr.Blocks() as demo:
    gr.Markdown("# Segment Image")
    gr.Markdown("Upload an image and let the model segment it.")
    
    with gr.Row():
        api_token = gr.Textbox(
            label="API Token", 
            placeholder="Enter your Hugging Face API token here"
        )
        model_name = gr.Textbox(
            label="AI Segmentation Model", 
            placeholder="Enter your Segmentation model here",
            value="facebook/mask2former-swin-tiny-coco-panoptic"
        )
    
    image_input = gr.Image(label="Upload Image")
    
    with gr.Row():
        api_token_validation = gr.Textbox(label="API Token Validation")
        segmented_image = gr.Image(label="Segmented Image")
        
    # New block for segments output
    
    with gr.Row():
        segments_output = gr.Textbox(label="Segments")
        
    examples = gr.Examples(
        examples=[
            ["Your HF API Token", "facebook/mask2former-swin-tiny-coco-panoptic", "https://upload.wikimedia.org/wikipedia/commons/7/74/A-Cat.jpg"]
        ],
        inputs=[api_token, model_name, image_input]
    )
    
    api_token.change(
        fn=segment_gradio_image, 
        inputs=[api_token, model_name, image_input], 
        outputs=[api_token_validation, segmented_image, segments_output]
    )

    model_name.change(
        fn=segment_gradio_image, 
        inputs=[api_token, model_name, image_input], 
        outputs=[api_token_validation, segmented_image, segments_output]
    )

    image_input.change(
        fn=segment_gradio_image, 
        inputs=[api_token, model_name, image_input], 
        outputs=[api_token_validation, segmented_image, segments_output]
    )

demo.launch()