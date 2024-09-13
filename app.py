import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import os

# Check the current operating system
plt = platform.system()

# Force paths to PosixPath if running on non-Windows systems
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath  # Redirect WindowsPath to PosixPath

# Streamlit app title
st.title('Transportni klassifikatsiya qiluvchi model')

# File uploader for images
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    # Create an image object that the model can process
    img = PILImage.create(file)

    # Convert the model path explicitly to PosixPath if not on Windows
    model_path = pathlib.Path('transport_model.pkl')
    if plt != 'Windows':
        model_path = model_path.as_posix()  # Convert the path to a POSIX string format

    # Load the model
    try:
        model = load_learner(model_path)  # Attempt to load the learner
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Predict the class of the uploaded image
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # Display the probability chart using Plotly
    fig = px.bar(y=model.dls.vocab, x=probs * 100)
    st.plotly_chart(fig)
