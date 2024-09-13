import streamlit as st
from fastai.vision.all import *
import pathlib
import platform

# Determine the current operating system
plt = platform.system()

# Set path handling based on the OS without redefining the classes
if plt == 'Linux' or plt == 'Darwin':  # Handles Linux and macOS
    temp = pathlib.PosixPath
else:
    temp = pathlib.WindowsPath

# Streamlit app title
st.title('Transportni klassifikatsiya qiluvchi model')

# File uploader for images
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    # Create an image for the model to predict
    img = PILImage.create(file)

    # Load the model (ensure the path is correct)
    model = load_learner('transport_model.pkl')

    # Predict the class of the uploaded image
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # Display the probability chart using Plotly
    fig = px.bar(y=model.dls.vocab, x=probs * 100)
    st.plotly_chart(fig)
