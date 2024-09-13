import streamlit as st
from fastai.vision.all import *
import pathlib
import platform

# Check the current operating system
plt = platform.system()

# Set path handling based on the OS without redefining internal classes
if plt == 'Linux' or plt == 'Darwin':  # Handles Linux and macOS systems
    # Force PosixPath if the system is not Windows
    pathlib.WindowsPath = pathlib.PosixPath

# Streamlit app title
st.title('Transportni klassifikatsiya qiluvchi model')

# File uploader for images
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    # Create an image object that the model can process
    img = PILImage.create(file)

    # Load the model ensuring correct path type
    model_path = pathlib.Path('transport_model.pkl')
    try:
        model = load_learner(model_path)
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
