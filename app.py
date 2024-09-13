import streamlit as st
from fastai.vision.all import *
import pathlib
import platform

# Set the correct path type based on the operating system
plt = platform.system()

# Streamlit app title
st.title('Transportni klassifikatsiya qiluvchi model')

# File uploader for images
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    # Create an image object that the model can process
    img = PILImage.create(file)

    # Load the model using pathlib.Path to handle paths appropriately
    model_path = pathlib.Path('transport_model.pkl')
    model = load_learner(model_path)  # Ensure model path is handled as a Path object

    # Predict the class of the uploaded image
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # Display the probability chart using Plotly
    fig = px.bar(y=model.dls.vocab, x=probs * 100)
    st.plotly_chart(fig)
