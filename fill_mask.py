import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModel, pipeline

nlp = pipeline("fill-mask")

st.title('Fill Mask')

with st.form(key='my_form'):
    text_input = st.text_input(label='Enter sentence')
    submit_button = st.form_submit_button(label='Submit')
  
    result = nlp(text_input)
              
    st.write(result)
