import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# App title
st.title("Fake News Detector")

# Description
st.write("Enter a News Article below to check whether it is Fake or Real.")

# Input text
inputn = st.text_area("News Article:", "")

# Button to check news
if st.button("Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is Real!")
        else:
            st.error("❌ The News is Fake!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
