import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path(__file__).parent.parent / "assets/Ubuntu-R.ttf"


def draw_bbox(image: np.ndarray, 
              box: list, 
              label: str = "", 
              color: tuple = (255, 0, 0), 
              text_color: tuple = (255, 255, 255)):
    lw = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, 
                  color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, 
                               fontScale=lw / 3, thickness=tf)[0] 
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, (*color, 1), -1, cv2.LINE_AA) 
        cv2.putText(image,
                    label, 
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    text_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return image


def draw_ocr_text(image: np.ndarray,
                  text: str = "",
                  color: tuple = (255, 0, 0),
                  text_color: tuple = (255, 255, 255)):
    # Increase the resolution of the given image
    img = Image.fromarray(image)
    scale_factor = 9 if img.size[0] < 30 else 3
    img = img.resize((x * scale_factor for x in img.size))
    img_width, img_height = img.size
    
    initial_font_size = 72  # Starting font size
    
    # Initial font load
    font = ImageFont.truetype(FONT_PATH, initial_font_size)
    
    # Try to fit the text within the image width
    text_width, text_height = font.getbbox(text)[2:]
    max_text_width = img_width * 0.9 

    while text_width > max_text_width:
        # Decrease font size to try to fit the text
        initial_font_size -= 2
        font = ImageFont.truetype(FONT_PATH, initial_font_size)
        text_width, text_height = font.getbbox(text)[2:]
        
        if initial_font_size <= 1:
            break

    font_scale_factor = img_width / 400
    rectangle_height = text_height + int(20 * font_scale_factor)
    
    img_with_rectangle = Image.new('RGB', 
                                   (img_width, 
                                    img_height + rectangle_height), 
                                    (0, 0, 0))
    img_with_rectangle.paste(img, (0, 0))
    
    # Draw the rectangle and text
    draw = ImageDraw.Draw(img_with_rectangle)
    draw.rectangle([0, img_height, img_width, 
                    img_height + rectangle_height], fill=color)
    text_x = (img_width - text_width) // 2
    text_y = img_height + (rectangle_height - text_height) // 2
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    return np.array(img_with_rectangle)
