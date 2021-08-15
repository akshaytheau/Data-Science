import streamlit as st1
from styleformer import Styleformer
import torch
sf1 = Styleformer(style = 3) 
st1.title('Passive Voice to Active Voice Converter')
st1.write("Please enter your sentence in passive voice")
text1 = st1.text_input('Entered Text')
if st1.button('Convert Passive to Active'):
  target_sentence1 = sf1.transfer(text1)
  st1.write(target_sentence1)
else:
     pass
