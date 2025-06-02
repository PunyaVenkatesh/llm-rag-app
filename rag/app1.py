import streamlit as st
from transformers import pipeline

st.title("Quick QA Test")
qa = pipeline("text2text-generation", model="google/flan-t5-small")
question = st.text_input("Ask me anything:")
if question:
    response = qa(question)[0]["generated_text"]
    st.write("Answer:", response)
