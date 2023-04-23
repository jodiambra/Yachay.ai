#import packages
import streamlit as st
import pandas as pd
import plotly_express as px
from PIL import Image
from streamlit.commands.page_config import Layout


#----------------------------#
# Upgrade streamlit library
# pip install --upgrade streamlit

#-----------------------------#
# Page layout
icon = Image.open('images/geo.ico')

st.set_page_config(page_title='Yachay.ai Externship',
                   page_icon=icon,
                   layout='wide',
                   initial_sidebar_state="auto",
                   menu_items=None)


st.title('Yachay.ai')
st.header('Text based geolocation prediction')

image1 = Image.open('images/tweet.png')
st.image(image1, use_column_width='auto')

st.write('Yachay is an open-source machine learning community with decades worth of natural language data from media,',
        'the dark web, legal proceedings, and government publications. They have cleaned and annotated the data, and',
        'created a geolocation detection tool. They are looking for developers interested in contributing and improving', 
        'on the project. We are given a dataset of tweets, and another dataset of coordinates, upon which we will create',
        'a neural network to predict coordinates from text. ')