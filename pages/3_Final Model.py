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


# final_df = pd.read_csv('processed data/final_df.csv')
# y_xlm_20_rand = pd.read_csv('processed data/y_xlm_20_rand.csv')
final_X_array = np.array(final_df).astype('float64')
final_y_array = np.array(y_xlm_20_rand).astype('float64')



