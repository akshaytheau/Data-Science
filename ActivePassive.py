import streamlit as st
import pandas as pd
from styleformer import Styleformer
import torch
sf = Styleformer(style = 2) 
st.title('Active Voice to Passive Voice Converter')
st.write("Please enter your sentence in active voice")
text = st.text_input('Entered Text')
if st.button('Convert Active to Passive'):
  target_sentence = sf.transfer(text)
  st.write(target_sentence)
else:
     pass


