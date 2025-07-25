import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📩 Email Spam Detector")
st.markdown("Type or paste your message below:")

user_input = st.text_area("Enter email/message text", height=150)

if st.button("Check If Spam"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("🚨 This message is **SPAM** (Scam Alert!)")
        else:
            st.success("✅ This message is **NOT Spam**")
