import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

model = tf.keras.models.load_model('vehicle_model.keras')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Vehicle Performance Detector')

origin = st.selectbox('Origin', options = [1, 2, 3])
cylinders = st.slider('Number of Cylinders', 3, 8)
displacement = st.number_input('Displacement', value = None, placeholder="Enter displacement", format = "%.1f", step = 0.1)
horsepower = st.number_input('Horsepower', value = None, placeholder="Enter horsepower", step = 1)
weight = st.number_input('Weight', value = None, placeholder="Enter weight", step = 1)
acceleration = st.number_input('Acceleration', value = None, placeholder="Enter acceleration", format = "%.1f", step = 0.1)
year = st.slider('Year', 1970, 1982)

input_data = pd.DataFrame({
    'origin': [origin],
    'cylinders': [cylinders],
    'displacement': [displacement],
    'horsepower': [horsepower],
    'weight': [weight],
    'acceleration': [acceleration],
    'year': [year]
})

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_val = prediction[0][0]

st.write(F"Kilometers per Liter: {prediction_val:.6f}")