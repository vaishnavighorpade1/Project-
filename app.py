import streamlit as st 
import numpy as np
import joblib

model = joblib.load("rf_model.pkl")

st.title("Laptop Prize Predictor")

st.divider()

st.write("Welcome to Laptop Prize Predictor!!💻. " \
"Select your configuration and get the prize👇")
st.divider()

processor_speed = st.number_input("Select processor speed", value = 2.50, step = 0.50)
ram_size = st.number_input("Select Ram Size", value = 16, step = 8)
storage_capacity = st.number_input("Select Storage Capacity", value = 512 , step = 256)

x = [processor_speed, ram_size , storage_capacity]

st.divider()

prediction = st.button("Prize Estimator Button")

st.divider()

if prediction:

    st.balloons()

    x1 = np.array(x)

    prediction = model.predict([x1])

    st.write(f"Prize estimation for laptop is {prediction[0]:.2f}")

else:
    st.write("Please use the button for getting a prediction")









