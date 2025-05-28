# rag/utils.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
import os
import pickle

# Set the embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_FILE = os.path.join(VECTORSTORE_DIR, "faiss_index")

def create_or_load_vectorstore(chunks):
    """Create or load a FAISS vector store from chunks."""
    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR)

    if os.path.exists(VECTORSTORE_FILE):
        with open(VECTORSTORE_FILE, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        vectorstore = FAISS.from_texts(chunks, embedding)
        with open(VECTORSTORE_FILE, "wb") as f:
            pickle.dump(vectorstore, f)

    return vectorstore

def retrieve_relevant_chunks(query, vectorstore, k=3):
    """Retrieve the top-k relevant chunks for a query using the vector store."""
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in docs]
