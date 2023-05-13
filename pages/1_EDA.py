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

st.title('Exploratory Data Analysis')
"---"

# read dataset
df_main = pd.read_csv('data/Main_Dataset.csv',
                      parse_dates=['timestamp'], index_col=['timestamp'])

# sort by timestamp
df_main.sort_index(inplace=True)

# load cluster data
df_cl = pd.read_csv('data/Clusters_Coordinates.csv')

# Making timestamp features


def make_features(data):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['week'] = data.index.isocalendar().week
    data['day'] = data.index.day
    data['day_of_week'] = data.index.day_of_week
    data['day_of_year'] = data.index.day_of_year
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute
    data['second'] = data.index.second


make_features(df_main)


# merge main and cluster coordinates
df = df_main.merge(df_cl, on='cluster_id', sort=True)

# drop missing values
df.dropna(inplace=True)


#------------------------------------------#

# EDA section
st.title('')
st.subheader('Exploratory Data Analysis')
columns = st.selectbox('Select Column', [
                       'cluster_id', 'month', 'week', 'day', 'day_of_week', 'day_of_year', 'hour', 'minute', 'second', 'user_id'])

@st.cache_data
def plot_hist(df, columns):
    fig = px.histogram(df[columns], title='Distribution of ' + str.upper(columns).replace(
        '_', ' '), labels={'value': str(columns).replace('_', ' ')}, height=800, width=1200)
    return fig


st.plotly_chart(plot_hist(df, columns), use_container_width=True)

"---"
#--------------------#
# Tweets dataframe
st.title('')
st.header('Tweets')

# looking through tweets
number = st.slider('Select Number of Tweets', 1, 1000, 10)

@st.cache_data
def tweet_lists(df, number):
    tweets = df.text.tolist()
    dataframe = tweets[:number]
    return dataframe

tweets = tweet_lists(df, number)
st.dataframe(tweets, width=1500)

"---"

