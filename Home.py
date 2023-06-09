#import packages
import requests
from streamlit_lottie import st_lottie
import streamlit as st
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
                   menu_items={'Get Help':'https://github.com/jodiambra/Yachay.ai',
                               'About': 'https://www.linkedin.com/in/jodiambra/'}
                    )


st.title('Yachay.ai')
"---"
st.header('Text based geolocation prediction')

# image1 = Image.open('images/tweet.png')
# st.image(image1, use_column_width='auto')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

twitter_bird = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_nkf5e15x.json')

st_lottie(twitter_bird, height=300, width=400, quality='high')

st.write('Yachay is an open-source machine learning community with decades worth of natural language data from media,',
        'the dark web, legal proceedings, and government publications. They have cleaned and annotated the data, and',
        'created a geolocation detection tool. They are looking for developers interested in contributing and improving', 
        'on the project. We are given a dataset of tweets, and another dataset of coordinates, upon which we will create',
        'a neural network to predict coordinates from text. ')


# Display the link in Streamlit
st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/jodiambra/Yachay.ai)", unsafe_allow_html=True)

