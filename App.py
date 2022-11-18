#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import streamlit as st
import numpy as np
from keras.models import load_model
savedModel=load_model('gru_modelll.h5')
savedModel.summary()

def main():
    st.title("Safaricom Shares Price Prediction Application")
    st.header('Enter the Day to Predict Price:')
    Day = st.number_input('Day Number:', min_value=0, max_value=2000, value=1)
    if st.button('Predict'):
        st.code(savedModel.predict(Day))
        st.success(f'The predicted Price of Shares is {prediction[0]:.2f}')
if __name__=='__main__': 
        main()




