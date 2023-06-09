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

st.title('Natural Language Processing Features')
st.subheader('')
# sentiment analysis
st.subheader('Sentiment Analysis')
sent = pd.read_csv('inputs/sent.csv', header=None)
sent_counts =sent[0].value_counts()
st.plotly_chart(px.bar(sent_counts, color=sent_counts.index,  title='Tweet Sentiment', height=600, width=800, 
        template='plotly_dark', labels={'value': 'Sentiment'}))

"---"

# Language detection
st.subheader('Language Detection')
language = pd.read_csv('inputs/lan.csv', header=None)
# counts of the different languages
lan_counts = language[0].value_counts()
st.plotly_chart(px.bar(lan_counts, color=lan_counts.index, title='Tweet Languages', height=800, width=1200, 
        template='plotly_white', labels={'index': 'Languages', 'value': 'Count'}), use_container_width=True)

"---"
# Topics analysis
st.subheader('Topics Analysis')
topics = pd.read_csv('inputs/topics.csv', header=None)
topic_counts= topics[0].value_counts()
st.plotly_chart(px.bar(topic_counts, color=topic_counts.index, title='Tweet Topics', height=800, width=1200, 
        template='plotly_dark', labels={'index': 'Topics', 'value': 'Count'}), use_container_width=True)

"---"

# Name Entity analysis
st.subheader('Name Entity Analysis')
entity = pd.read_csv('inputs/ner.csv', header=None)
entity_counts= entity[0].value_counts()
st.plotly_chart(px.bar(entity_counts, color=entity_counts.index, title='Tweet Entities', height=800, width=1200, 
        template='plotly_dark', labels={'index': 'Entities', 'value': 'Count'}), use_container_width=True)

"---"