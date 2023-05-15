#import packages
import streamlit as st
import pandas as pd
import plotly_express as px
from PIL import Image
from streamlit.commands.page_config import Layout
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
import math
from sklearn.metrics.pairwise import haversine_distances
from streamlit.components.v1 import components


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

st.title('Model Architecture')

"---"
#--------------------------#
# NLP section
st.title('')
st.header('Natural Language Processing')

st.write('We used a few pre-trained models to process our text into embeddings: BERT base uncased, BERT base multilingual uncased,', 
        'and XLM roberta large. These models each have native support for many languages, but XLM performed best. Due to the time and', 
        'resource intensity of processing text, we have provided a sample of 1,000 rows of pre-processed text with the various models.')
st.subheader('')

"---"

#---------------------------# 
# button columns for loading specific modeled data
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('BERT Base')
    #base = st.button('Base')
with col2:
    st.subheader('BERT Base Multilingual')
    #multi = st.button('Multi')
with col3:
    st.subheader('XLM Roberta Large')
    #xlm = st.button('XLM')

nlp = st.selectbox('pick nlp', ['base', 'multi', 'xlm'])

y = pd.read_csv('inputs/y.csv')

@st.cache_resource
def load_dataset(path):
    df = pd.read_csv(path)
    return df

if nlp=='base':
    X_base = load_dataset('inputs/X_base.csv')
    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=19)
    st.success('You loaded the BERT Base features and test set', icon='👊')
elif nlp=='multi':
    X_multi = load_dataset('inputs/X_multi.csv')
    X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=19)
    st.success('You loaded the BERT Base Multilingual features and test set', icon='👌')
elif nlp=='xlm':
    X_xlm = load_dataset('inputs/X_xlm.csv')
    X_train, X_test, y_train, y_test = train_test_split(X_xlm, y, test_size=0.2, random_state=19) 
    st.success('You loaded the XLM Roberta large features and test set', icon='👏')
else:
    st.error('You did not load a processed dataset yet. Click a button above.', icon='👆')

#---------------------------# 

# Loss function

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

#---------------------------#
"---"
# model section

st.title('')
st.header('Artificial Neural Network')


st.subheader('')
first_layer = st.text_input('Hidden neurons in first layer', 2000)
second_layer = st.text_input('Hidden neurons in second layer', 1000) 
epoch = st.select_slider('Select the number of epochs', [10, 50, 100, 200])

run_models = st.button('Ready to Run', use_container_width=True)


if run_models:
    tf.random.set_seed(19)
    optimizer = Adam(learning_rate=.0001)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.0000001)

    # define the model architecture
    model = Sequential()
    model.add(Dense(first_layer, activation='relu', input_dim=(X_train.shape[1])))
    model.add(Dense(second_layer, activation='relu'))
    model.add(Dense(2)) # output layer with 2 units for latitude and longitude

    # compile the model
    model.compile(optimizer=optimizer, loss=loss_haversine, metrics=['mse'])

    # train the model
    with tf.device('/GPU:0'):
        with st.spinner('Model is running.... Please be patient.'):
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.10, callbacks=[callback, reduce_lr])
        st.success('Model is done training.')
    # Convert the model history to a pandas DataFrame
    df_his = pd.DataFrame(history.history)

    # Create separate figures for loss and accuracy
    fig_loss = px.line(df_his, x=df_his.index, y=['loss', 'val_loss'], labels={'value': 'Loss', 'index': 'Epoch'}, title='Model Loss', template='plotly_dark')
    fig_acc = px.line(df_his, x=df_his.index, y=['mse', 'val_mse'], labels={'value': 'MSE', 'index': 'Epoch'}, title='Model MSE', template='plotly_dark')
    fig_lr = px.line(df_his, x=df_his.index, y='lr', labels={'value': 'Learning Rate', 'index': 'Epoch'}, title='Model Learning Rate', log_y=True, template='plotly_dark', width=300)

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
    col7.metric('Haversine Loss', round(model.evaluate(X_test, y_test)[0], 2))
    col8.metric('MSE', round(model.evaluate(X_test, y_test)[1], 2))



    # X test predictions
    preds = model.predict(X_test)   

    preds_df = pd.DataFrame(preds, columns=['lat_p', 'lng_p'])

    # y test dataframe
    y_df = y_test.reset_index(drop=True)

    # concat test and prediction coordinates
    coords = pd.concat([y_df, preds_df], axis=1)

    # convert test set coordinates to radians   
    y_test_rad = y_test * (math.pi/180)

    # convert prediction coordinates to radians
    preds_rad = preds * (math.pi/180)

    # calculate distance
    distances = haversine_distances(y_test_rad, preds_rad)[0]
    distances_km = distances * (6371000/1000)
    distances_mi = distances_km * 0.621371

    st.title('')
    tab1, tab2= st.tabs(['Kilometers', 'Miles'])
    with tab1:
        st.subheader('Kilometers')
        st.write(px.bar(distances_km, title='Distances Between Actual and Prediction', labels={'value': 'Distance (Km)'}, 
            template='plotly_dark', width=1300, height=800))

        st.write(px.box(distances_km, title='Distribution of Distances', labels={'value': 'Distance (Km)'}, template='plotly_white', width=900, height=700))
    with tab2:
        st.subheader('Miles')
        st.write(px.bar(distances_mi, title='Distances Between Actual and Prediction', labels={'value': 'Distance (mi)'}, 
                template='plotly_dark', width=1300, height=800))

        st.write(px.box(distances_mi, title='Distribution of Distances', labels={'value': 'Distance (mi)'}, template='plotly_white', width=900, height=700))
  
else:
    st.warning('Pick an NLP model first, change parameters, then click run', icon='🏃‍♂️') 

