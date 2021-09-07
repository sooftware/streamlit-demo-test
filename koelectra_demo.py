import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModel, pipeline

nlp = pipeline("fill-mask")

st.title('Fill Mask')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = AutoModel.from_pretrained('monologg/koelectra-base-v3-discriminator')

with st.form(key='my_form'):
    text_input = st.text_input(label='Enter sentence')
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
      with torch.no_grad():
  
        result = nlp(
          text_input, 
          model='monologg/koelectra-base-v3-discriminator',
          tokenizer='monologg/koelectra-base-v3-discriminator')
              
        st.write(result)
