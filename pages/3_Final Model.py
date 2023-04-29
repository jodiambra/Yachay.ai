#import packages
import numpy as np
from numpy import genfromtxt
import streamlit as st
import pandas as pd
import plotly_express as px
from PIL import Image
from streamlit.commands.page_config import Layout
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
import torch
import transformers
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import math
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from streamlit.components.v1 import components
import io 
import json
import requests
from streamlit_lottie import st_lottie

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
st.subheader('Final Model')


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

tweet = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_UAN5ABS6cI.json')

st_lottie(tweet, height=600, width=900, quality='high')
