import json
import requests
from streamlit_lottie import st_lottie
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
st.header('Abstracts')

st.subheader("")

#--------------------#

st.subheader('An Effective Approach for Geolocation Prediction In Twitter Streams Using Clustering Based Discretization')

st.write("Micro-blogging services, such as Twitter, have provided an indis-",
        "pensable channel to communicate, access, and exchange current affairs. Understanding the", 
        "dynamics of users behavior and their geographical location is",
        "key to providing services such as event detection, geo-aware recommendation",
        "and local search. The geographical location prediction problem we address",
        "is to predict the geolocation of a user based on textual tweets. In this paper,",
        "we develop a clustering based discretization approach which is an effective",
        "combination of three well-known machine learning algorithms, e.g. K-means",
        "clustering, support vector machines, and K-nearest neighbor, to tackle the task",
        "of geolocation prediction in Twitter streams. Our empirical results indicate that",
        "our approach outperforms previous attempts on a publicly available dataset and",
        "that it achieves state-of-the-art performance.")

with open("literature/An Effective Approach for Geolocation Prediction.pdf", "rb") as file1:
    btn = st.download_button(
            label=":green[Download Article]",
            data=file1,
            file_name="An Effective Approach for Geolocation Prediction.pdf",
            mime="text/csv"
          )

#--------------------#
st.divider()
#--------------------#

st.subheader('Location Prediction with Communities in User Ego-Net in Social Media')

st.write("Social media embed rich but noisy signals of physical",
        "locations of their users. Accurately inferring a user’s location can",
        "significantly improve the user’s experience on the social media",
        "and enable the development of new location-based applications.",
        "This paper proposes a novel community-based approach for",
        "predicting the location of a user by using communities in the ego-net",
        "of the user. We further propose both geographical proximity",
        "and structural proximity metrics to profile communities in the",
        "ego-net of a user, and then evaluate the effectiveness of each",
        "individual metric on real social media data. We discover that",
        "geographical proximity metrics, such as average/median haversine",
        "distance and community closeness, are strong indicators of a good",
        "community for geotagging. In addition, structural proximity metric",
        "conductance performs comparable to geographical proximity",
        "metrics while triangle participation ratio and internal density are",
        "weak location indicators. To the best of our knowledge, this is",
        "the first effort to infer the physical location of a user from the",
        "perspective of latent communities in the user’s ego-net.")

with open("literature/Location Prediction with Communities in User.pdf", "rb") as file2:
    btn = st.download_button(
            label=":green[Download Article]",
            data=file2,
            file_name="Location Prediction with Communities in User.pdf",
            mime="text/csv"
          )
    
#--------------------#   
st.divider()
#--------------------#

st.subheader("Multiview Deep Learning for Predicting Twitter User's Location")

st.write("The problem of predicting the location of users on large social networks like",
        "Twitter has emerged from real-life applications uch as social unrest detection and",
        "online marketing. Twitter user geolocation is a difficult and active research topic",
        "with a vast literature. Most of the proposed methods follow either a content-based or",
        "a network-based approach. The former exploits user-generated content while the latter",
        "utilizes the connection or interaction between Twitter users. In this paper, we introduce",
        "a novel method combining the strength of both approaches. Concretely, we propose a",
        "multi-entry neural network architecture named MENET leveraging the advances in deep learning",
        "and multiview learning. The generalizability of MENET enables the integration of multiple",
        "data representations. In the context of Twitter user geolocation, we realize MENET with",
        "textual, network, and metadata features. Considering the natural distribution of Twitter users",
        "across the concerned geographical area, we subdivide the surface of the earth into multi-scale",
        "cells and train MENET with the labels of the cells. We show that our method outperforms the state",
        "of the art by a large margin on three benchmark datasets.")

with open("literature/Multiview Deep Learning for Predicting Twitter.pdf", "rb") as file3:
    btn = st.download_button(
            label=":green[Download Article]",
            data=file3,
            file_name="Multiview Deep Learning for Predicting Twitter.pdf",
            mime="text/csv"
          )
    
#--------------------#   
st.divider()
#--------------------#

st.subheader("Predicting the Geolocation of Tweets Using BERT-Based Models Trained on Customized Data")

st.write("This research is aimed to solve the tweet/user geolocation prediction task and",
        "provide a flexible methodology for the geotagging of textual big data. The suggested",
        "approach implements neural networks for natural language processing (NLP) to estimate the",
        "location as coordinate pairs (longitude, latitude) and two-dimensional Gaussian Mixture",
        "Models (GMMs). The scope of proposed models has been finetuned on a Twitter dataset",
        "using pretrained Bidirectional Encoder Representations from Transformers (BERT) as base",
        "models. Performance metrics show a median error of fewer than 30 km on a worldwide-",
        "level, and fewer than 15 km on the US-level datasets for the models trained and evaluated",
        "on text features of tweets’ content and metadata context.")

with open("literature\Predicting the Geolocation of.pdf", "rb") as file3:
    btn = st.download_button(
            label=":green[Download Article]",
            data=file3,
            file_name="Predicting the Geolocation of.pdf",
            mime="text/csv"
          )
    
#--------------------#   
st.divider()
#--------------------#

st.subheader("Text-based Geolocation Prediction of Social Media Users with Neural Networks")

st.write("Inferring the location of a user has been a valuable",
        "step for many applications that leverage social media, such as",
        "marketing, security monitoring and recommendation systems.",
        "Motivated by the recent success of Deep Learning techniques",
        "for many other tasks such as computer vision, speech recognition,",
        "and natural language processing, we study the application",
        "of neural networks to the problem of geolocation prediction",
        "and experiment with multiple techniques to improve neural",
        "networks for geolocation inference based solely on text. Experimental",
        "results on three Twitter datasets suggest that choosing appropriate network",
        "architecture, activation function, and performing Batch Normalization,",
        "can all increase performance on this task.")

with open("literature/Text-based Geolocation Prediction of Social Media Users with Neural Networks.pdf", "rb") as file4:
    btn = st.download_button(
            label=":green[Download Article]",
            data=file4,
            file_name="Text-based Geolocation Prediction of Social Media Users with Neural Networks.pdf",
            mime="text/csv"
          )