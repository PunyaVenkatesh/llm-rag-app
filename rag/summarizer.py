from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Device set to {'cuda' if device == 0 else 'cpu'}")

# Faster summarization model
summarizer = pipeline("summarization", model="t5-small", device=device)

def summarize_text(text_chunks):
    summaries = []
    for chunk in text_chunks:
        if len(chunk.strip().split()) > 10:
            try:
                summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summary = chunk
        else:
            summary = chunk
        summaries.append(summary)
    return summaries
