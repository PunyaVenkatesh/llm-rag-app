import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from rag.loader import load_document, chunk_text
from rag.summarizer import summarize_text
from rag.qa import get_qa_chain

st.set_page_config(page_title="Research Paper Summarizer and Q&A", layout="wide")
st.title("Research Paper Summarizer and Q&A")

uploaded_file = st.file_uploader("Upload your research paper (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    with st.spinner("Reading document..."):
        raw_text = load_document(uploaded_file)
        st.success("Document loaded successfully!")

    st.subheader("Document Preview")
    st.text_area("Raw Content", raw_text[:3000] + "..." if len(raw_text) > 3000 else raw_text, height=300)

    @st.cache_data
    def get_summaries(text):
        chunks = chunk_text(text, max_tokens=1000)
        return summarize_text(chunks)

    with st.spinner("Summarizing..."):
        summaries = get_summaries(raw_text)

    st.subheader("Structured Summary")
    for i, summary in enumerate(summaries, start=1):
        st.markdown(f"**Section {i}:** {summary}")

    st.subheader("Ask Questions About the Paper")
    question = st.text_input("Enter your question")

    @st.cache_resource
    def load_qa_chain(text):
        return get_qa_chain(text)

    if question:
        with st.spinner("Finding answer..."):
            qa_chain = load_qa_chain(raw_text)
            answer = qa_chain.run(question)
            st.success("Answer:")
            st.write(answer)
