import json, os, uuid
from dotenv import load_dotenv
import chromadb
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# persistent local DB under ./chroma
chroma = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma"))
collection = chroma.get_or_create_collection(name="book_summaries")


def embed(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def main():
    with open("data/book_summaries.json", "r", encoding="utf-8") as f:
        books = json.load(f)

    ids, embeddings, documents, metadatas = [], [], [], []
    for b in books:
        themes_str = ", ".join(b["themes"])  # <- make it a string
        doc = (
            f"Title: {b['title']}\n"
            f"Summary: {b['summary']}\n"
            f"Themes: {themes_str}"
        )
        ids.append(str(uuid.uuid4()))
        documents.append(doc)
        # metadata values must be primitives
        metadatas.append({"title": b["title"], "themes": themes_str})
        embeddings.append(embed(doc))


    # add with precomputed embeddings (so we control the embedding model)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Indexed {len(ids)} books into Chroma.")


if __name__ == "__main__":
    main()
