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
from segmentation_utils import segment_and_overlay_results
from icecream import ic

# Disable the prefix
ic.configureOutput(prefix='')

ic("--- Setting up environment ---")
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#ic("--- Calling segment_and_overlay_results with local file ---")
text_image, segments = segment_and_overlay_results('cats.jpg','facebook/mask2former-swin-tiny-coco-panoptic',HUGGINGFACEHUB_API_TOKEN)
text_image.show()

#ic("--- Calling segment_and_overlay_results with URL ---")
text_image, segments = segment_and_overlay_results('https://upload.wikimedia.org/wikipedia/commons/7/74/A-Cat.jpg','facebook/mask2former-swin-tiny-coco-panoptic',HUGGINGFACEHUB_API_TOKEN)
text_image.show()

ic("--- Calling segment_and_overlay_results with numpy image array ---")
text_image, segments = segment_and_overlay_results(cv2.imread('cats.jpg'),'facebook/mask2former-swin-tiny-coco-panoptic',HUGGINGFACEHUB_API_TOKEN)
text_image.show()

