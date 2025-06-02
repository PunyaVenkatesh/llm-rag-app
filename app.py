import os
import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_pdf
from rag.summarizer import summarize_text
from rag.qa import qa_bot  # returns run_qa() function

st.set_page_config(page_title=" Research Summarizer & QA", layout="wide")
load_dotenv()

st.title("Research Paper Summarizer & Q&A")
st.markdown("Upload a research paper (PDF) to get a structured summary and ask questions using LLMs.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading and extracting text from PDF..."):
        pdf_text = load_pdf(uploaded_file)

    if not pdf_text.strip():
        st.error(" The PDF appears empty or unreadable.")
    else:
        # Summary Section
        if st.button(" Generate Summary"):
            st.info(" Summarizing the paper. This may take a moment...")
            try:
                summary = summarize_text(pdf_text)
                st.success("Summary Generated!")

                st.subheader(" Introduction")
                st.write(summary.get("Introduction", "Not available."))

                st.subheader(" Main Points")
                st.write(summary.get("Main Points", "Not available."))

                st.subheader(" Conclusion")
                st.write(summary.get("Conclusion", "Not available."))

            except Exception as e:
                st.error(f" Summarization failed: {str(e)}")

        # Question Answering Section
        st.markdown("---")
        st.subheader("Ask a Question About the Paper")

        query = st.text_input("Type your question here...")

        if query:
            with st.spinner("Searching and answering..."):
                try:
                    run_qa = qa_bot(pdf_text)  #  Updated to return function
                    answer = run_qa(query)     #  Execute the function with the user's question
                    st.success("Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f" QA failed: {str(e)}")
