import streamlit as st
import pandas as pd
from styleformer import Styleformer
import torch
sf = Styleformer(style = 0) 
st.title('Casual to Formal converter')
st.write("Please enter your casual text")
text = st.text_input('Enter some text')
if st.button('Hit me'):
  target_sentence = sf.transfer(text)
  st.write(target_sentence)
else:
     pass


