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
    st.title("Safaricom shares Price Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Price of shares Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Day= st.number_input('Day:', min_value=1, max_value=2000, value=1)
    result=""
    if st.button("Predict"):
        result=savedModel.predict(Day)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()


# In[ ]:




