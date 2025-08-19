import json

with open("data/book_summaries.json", "r", encoding="utf-8") as f:
    _BOOKS = {b["title"]: b for b in json.load(f)}

def get_summary_by_title(title: str) -> str:
    b = _BOOKS.get(title)
    return b["summary"] if b else "Summary not found."
