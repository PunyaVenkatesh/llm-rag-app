from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document

from transformers import pipeline

def get_qa_chain(raw_text):
    # 1. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(raw_text)
    docs = [Document(page_content=t) for t in texts]

    # 2. Use local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # 3. Use HuggingFace Transformers pipeline
    hf_pipeline = pipeline(
        "text2text-generation", 
        model="google/flan-t5-small", 
        max_length=256
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 4. Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
