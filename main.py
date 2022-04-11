import streamlit as st
from PIL import Image
import pytesseract


def load_image(img_file):
    # Read in and make greyscale
    img = Image.open(img_file)
    return img

st.title('1506 Help')

image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

if image_file is not None:
    img = load_image(image_file)
    text = pytesseract.image_to_string(img)

    st.write(text)