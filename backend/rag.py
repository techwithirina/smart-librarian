import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma"))
collection = chroma.get_or_create_collection(name="book_summaries")


def embed_query(text: str):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


def search_books(user_query: str, k: int = 3):
    q_emb = embed_query(user_query)
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    # columnar shape → take first query
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"title": meta["title"], "themes": meta["themes"], "document": doc, "distance": dist})
    return hits
