import streamlit as st
from PIL import Image
from img_process import process


def load_image_gray(img_file):
    # Read in and make greyscale
    img = Image.open(img_file).convert('L')
    return img


def load_image(img_file):
    # Read in and make greyscale
    img = Image.open(img_file)
    return img


st.title('1506 Help')

image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

if image_file is not None:

    st.image(load_image(image_file), width=250)
    pil = load_image_gray(image_file)

    processed_data = process(pil)
    st.dataframe(processed_data)