import base64
import io
import logging
import os
from PIL import Image
import numpy as np
from typing import Union, Tuple, Optional

logger = logging.getLogger(__name__)

def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    width, height = image.size
    max_width, max_height = max_size
    
    if width <= max_width and height <= max_height:
        return image
    
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != "RGB":
        return image.convert("RGB")
    return image


def image_to_base64(image: Union[str, Image.Image]) -> str:
    if isinstance(image, str):
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    else:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def preprocess_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> Optional[str]:
    try:
        image = Image.open(image_path)
        image = convert_to_rgb(image)
        image = resize_image(image, max_size)
        return image_to_base64(image)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None