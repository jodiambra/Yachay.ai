#import packages
import streamlit as st
import pandas as pd
import plotly_express as px
from PIL import Image
from streamlit.commands.page_config import Layout
import numpy as np 
from statistics import mode
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

@st.cache_data
def load(path):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def plot(chart):
    return st.plotly_chart(chart, use_container_width=True)

coords = load('data/Clusters_Coordinates.csv')

st.title('Maps')
st.subheader('')
st.subheader('Cluster of Tweets')
plot(px.scatter_geo(coords, lat='lat', lon='lng', width=1400, height=900, opacity=.6, template='presentation',
                               scope='north america', projection='stereographic'))

#----------------#
st.divider()
#----------------#

st.subheader('Language Clusters')

# df merge dataframe with features and coordinates
df_merge = load('processed data/df_merge.csv')
id_encoder = LabelEncoder()
df_merge['user_id'] = id_encoder.fit_transform(df_merge[['user_id']])
lang_mode = df_merge.groupby('language')[['lat', 'lng']].agg(mode)
lang_mode.reset_index(inplace=True)

plot(px.scatter_geo(lang_mode, color='language', lat='lat', lon='lng', width=1400, height=900, 
                               template='presentation', opacity=.6, scope='north america', projection='stereographic') 
                              )

#----------------#
st.divider()
#----------------#

st.subheader('Distribution of Languages')

# plot(px.scatter_geo(df_merge, color='language', lat='lat', lon='lng', width=1400, template='presentation',
                    # height=900, opacity=.6, scope='north america', projection='stereographic'))


#----------------#
st.divider()
#----------------#
st.subheader('Distribution of Topics')
plot(px.scatter_geo(df_merge, color='topic', lat='lat', lon='lng', width=1400, height=900, template='presentation',
                    opacity=.6, scope='north america', projection='stereographic'))