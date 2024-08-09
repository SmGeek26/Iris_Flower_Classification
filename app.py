import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('iris_model.pkl')

# Define function to make predictions
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app
st.title("Iris Flower Classification")

st.write("Enter the details of the Iris flower:")

sepal_length = st.number_input("Sepal Length", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width", min_value=0.0, format="%.2f")

if st.button("Classify"):
    prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f"The predicted class is: {prediction}")