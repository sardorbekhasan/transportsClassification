import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title('Transportni klassifikatsiya qiluvchi model')

file = st.file_uploader('Rasm yuklash', type=['png','jpeg', 'gif', 'svg'])
if file:
    img = PILImage.create(file)
    st.image(file)
    
    model = load_learner('transport_model.pkl')
    pred, pred_id, probs = model.predict(img)
    
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100:.1f}%")
    fig = px.bar(y=model.dls.vocab, x=probs*100)    
    st.plotly_chart(fig)
    
