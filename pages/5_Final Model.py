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
from tensorflow.keras.models import load_model

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

# lottie Animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

tweet = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_UAN5ABS6cI.json')

st_lottie(tweet, height=600, width=900, quality='high')


#----------------------------------------------------------#

# load model
@st.cache_resource
def load_models(path):
    model = load_model(path)
    return model

# haversine distance loss
RADIUS_KM = 6378.1

def degrees_to_radians(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180

def loss_haversine(observation, prediction):    
    obv_rad = tf.map_fn(degrees_to_radians, observation)
    prev_rad = tf.map_fn(degrees_to_radians, prediction)

    dlon_dlat = obv_rad - prev_rad 
    v = dlon_dlat / 2
    v = tf.sin(v)
    v = v**2

    a = v[:,1] + tf.cos(obv_rad[:,1]) * tf.cos(prev_rad[:,1]) * v[:,0] 

    c = tf.sqrt(a)
    c = 2* tf.math.asin(c)
    c = c*RADIUS_KM
    final = tf.reduce_sum(c)

    #if you're interested in having MAE with the haversine distance in KM
    #uncomment the following line
    final = final/tf.dtypes.cast(tf.shape(observation)[0], dtype= tf.float32)

    return final
#------------------------------------------------#
# load features and target
X = pd.read_csv('inputs/x_merged.csv')
y = pd.read_csv('inputs/y.csv')

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=19) # split 20% of data to make validation set

#------------------------------------------------#
st.divider()

st.subheader('Final Model')
# load Final model
final_model = st.button('Load the Final Model')

if final_model:
    # register the custom loss function
    tf.keras.utils.get_custom_objects()['loss_haversine'] = loss_haversine
    
    with st.spinner('Model is Loading.... Please be patient.'):
        model = load_models('models/xlm_merge_keras')
    


    # load history
    st.success('Model is Loaded')
    history = pd.read_csv('history/history.csv')

    # Create separate figures for loss and accuracy
    fig_loss = px.line(history, x=history.index, y=['loss', 'val_loss'], labels={'value': 'Loss', 'index': 'Epoch'}, title='Model Loss', template='plotly_dark')
    fig_acc = px.line(history, x=history.index, y=['mse', 'val_mse'], labels={'value': 'MSE', 'index': 'Epoch'}, title='Model MSE', template='plotly_dark')
    fig_lr = px.line(history, x=history.index, y='lr', labels={'value': 'Learning Rate', 'index': 'Epoch'}, title='Model Learning Rate', log_y=True, template='plotly_dark', width=300)

    # Show the figures
    col4, col5, col6 = st.columns(3)
    with col4:
        fig_loss
    with col5:
        fig_acc
    with col6:
        fig_lr

    # evaluation on test set

    st.subheader('Model Evaluation on Test Set: ')
    col7, col8 = st.columns(2)
    with col7:
        col7.metric('Haversine Loss', 1414.400)
    with col8:
        col8.metric('MSE', 158.084)

    



