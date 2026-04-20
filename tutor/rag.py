from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    open("rag_data/math.md", "r", encoding="utf-8").read(),
    open("rag_data/algebra.md", "r", encoding="utf-8").read()
]

doc_embeddings = model.encode(docs)

def search(query):
    q_emb = model.encode([query])

    # cosine similarity через dot product
    scores = np.dot(doc_embeddings, q_emb.T)

    best_idx = int(np.argmax(scores))

    return docs[best_idx]