import cv2
import streamlit
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import pyocr
import pyocr.builders


def process(img):
    tools = pyocr.get_available_tools()
    tool = tools[0]

    cv = pil_to_cv(img)
    # Line-less copy
    image_without_borders = remove_lines(cv)
    # Get results
    cropped_without_borders = crop_periods_of_service(image_without_borders)
    # cropped_with_borders = crop_periods_of_service(img)

    cropped_without_borders = cv_to_pil(cropped_without_borders)
    # cropped_with_borders = cv_to_pil(cropped_with_borders)
    return cropped_without_borders


def pil_to_cv(img):
    # For reversing the operation:
    im_np = np.asarray(img)
    return im_np


def cv_to_pil(img):
    # You may need to convert the color.
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
    return im_pil


def remove_lines(form_image):
    # Thresholding the image
    (thresh, img_bin) = cv2.threshold(form_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # (thresh, img_bin) = cv2.threshold(form_image, 128, 255)
    # Invert the image
    img_bin = 255 - img_bin

    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1] // 80

    # Verticle kernel of (1 X kernel_length)
    # > detects all the verticle lines from the image.

    # Horizontal kernel of (kernel_length X 1)
    # > detects all the horizontal line from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight
    # parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    invert = cv2.bitwise_not(img_final_bin)
    dst = cv2.addWeighted(form_image, 1, invert, 1, 0)

    return dst


def has_text(df, txt):
    return len(df[df['text'] == txt]) > 0


def crop_periods_of_service(img):
    # Get results
    df = pytesseract.image_to_data(img, lang='eng', output_type='data.frame')

    # Get width and height for crop
    height, width = img.shape

    if has_text(df, 'IDENTIFICATION'):
        y = df[df['text'] == 'IDENTIFICATION']['top'].iloc[0]
        h = df[df['text'] == 'IDENTIFICATION']['height'].iloc[0]
        finder = img[y + h:height, 0:width]

    elif has_text(df, 'NUMBER'):
        y = df[df['text'] == 'NUMBER']['top'].iloc[0]
        h = df[df['text'] == 'NUMBER']['height'].iloc[0]
        finder = img[y + h:height, 0:width]

    df = pytesseract.image_to_data(img, lang='eng', output_type='data.frame')

    if has_text(df, 'PERIODS'):
        y = df[df['text'] == 'PERIODS']['top'].iloc[0]
        h = df[df['text'] == 'PERIODS']['height'].iloc[0]
        finder = img[y + int(h * 6.5):y + int(h * 70), 0:width]

    elif has_text(df, 'SERVICE'):
        y = df[df['text'] == 'SERVICE']['top'].iloc[0]
        h = df[df['text'] == 'SERVICE']['height'].iloc[0]
        finder = img[y + int(h * 6.5):y + int(h * 70), 0:width]

    return finder

