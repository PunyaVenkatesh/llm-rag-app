import os
import hashlib
import json
from dotenv import load_dotenv
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from multiprocessing import Pool

load_dotenv()

# ✅ Use a fast summarization model
summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization",  # much faster than distilbart
    tokenizer="Falconsai/text_summarization",
    device=0 if os.getenv("USE_GPU") == "1" else -1
)

# ✅ Efficient chunking
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    return splitter.split_text(text)

# ✅ Summarize one chunk (for use in multiprocessing)
def summarize_chunk(chunk):
    try:
        trimmed = chunk[:1000]  # truncate to avoid long inputs
        result = summarizer(trimmed, max_length=80, min_length=25, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        return f"[Error] {str(e)}"

# ✅ Optional cache directory
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ✅ Generate a cache key for the full document
def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# ✅ Main summarization function
def summarize_text(raw_text: str) -> dict:
    if not raw_text.strip():
        return {
            "Introduction": "No input text provided.",
            "Main Points": "No input text provided.",
            "Conclusion": "No input text provided."
        }

    cache_key = get_cache_key(raw_text)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # Check cache
    if os.path.exists(cache_path):
        return json.load(open(cache_path))

    chunks = chunk_text(raw_text)
    print(f"🧩 Summarizing {len(chunks)} chunks...")

    # ✅ Use multiprocessing for parallel summarization
    with Pool(processes=4) as pool:
        summaries = pool.map(summarize_chunk, chunks)

    result = {
        "Introduction": summaries[0] if len(summaries) > 0 else "Not available",
        "Main Points": "\n\n".join(summaries[1:-1]) if len(summaries) > 2 else "Not enough content",
        "Conclusion": summaries[-1] if len(summaries) > 1 else "Not available"
    }

    # ✅ Cache the result
    with open(cache_path, "w") as f:
        json.dump(result, f)

    return result
