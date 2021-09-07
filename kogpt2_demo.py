import streamlit as st
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from transformers import ElectraModel, AutoConfig, GPT2LMHeadModel
from transformers.activations import get_activation
from transformers import AutoTokenizer


st.title('KoGPT2 Demo')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

with st.form(key='my_form'):
    text_input = st.text_input(label='Enter sentence')
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
      with torch.no_grad():
        inputs = tokenizer.encode(text_input)
        gen_ids = model.generate(torch.tensor([inputs]),
                           max_length=128,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)
        generated = tokenizer.decode(gen_ids[0,:].tolist())
              
        st.write(generated)
