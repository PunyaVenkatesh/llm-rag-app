import os
import hashlib
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

load_dotenv()

# Config
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = ".faiss_cache"
SMART_MODE = True  # 🔁 Set to False to use the faster model

QA_MODEL = (
    "google/flan-t5-base" if SMART_MODE else "google/flan-t5-small"
)

os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_vectorstore(texts, cache_key):
    cache_path = os.path.join(CACHE_DIR, cache_key)
    if os.path.exists(os.path.join(cache_path, "index.faiss")):
        return FAISS.load_local(
    folder_path=cache_path,
    embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
    allow_dangerous_deserialization=True  # ✅ TRUSTED LOCAL CACHE
)

    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_texts(texts, embeddings)
    db.save_local(cache_path)
    return db

def qa_bot(text: str):
    # Step 1: Split document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_text(text)

    # Step 2: Build or load cached vector store
    cache_key = get_cache_key("".join(texts))
    db = get_cached_vectorstore(texts, cache_key)

    # Step 3: Set up retriever with more chunks (k=5)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Optional: Show retrieved context
    print("\n🔍 Retrieved Chunks for Debugging:")
    sample_docs = retriever.get_relevant_documents("What is the main innovation introduced by the Transformer?")
    for i, doc in enumerate(sample_docs):
        print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:400]}")

    # Step 4: Load model (fast or smart)
    hf_pipe = pipeline(
        "text2text-generation",
        model=QA_MODEL,
        max_length=256,
        truncation=True
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # Step 5: Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    # Step 6: Custom wrapper to catch bad outputs
    def run_qa(query: str):
        answer = qa_chain.run(query)
        if answer.strip().startswith("[") or answer.strip().endswith("]") or len(answer.strip()) < 10:
            return "⚠️ The model could not generate a meaningful answer. Try rephrasing the question or increasing model size."
        return answer

    return run_qa
