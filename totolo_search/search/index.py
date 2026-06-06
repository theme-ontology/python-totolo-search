import json
import logging
import re
import shutil
import totolo
from pathlib import Path
import numpy as np
import tantivy

log = logging.getLogger(__name__)

DATA_DIR = Path.home() / ".totolo_search"
INDEX_DIR = DATA_DIR / "index"
DOCS_FILE = DATA_DIR / "docs.json"
PIECES_FILE = DATA_DIR / "pieces.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "it", "its", "that", "this",
    "was", "are", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "not", "no",
    "so", "then", "when", "where", "which", "who", "what", "how", "if",
    "i", "we", "he", "she", "they", "you", "me", "him", "her", "us", "them",
}


def _schema():
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("doc_id", stored=True)
    builder.add_text_field("search_text", stored=False, tokenizer_name="en_stem")
    return builder.build()


def _to_str(val) -> str:
    if not val:
        return ""
    if isinstance(val, (list, tuple)):
        return " ".join(str(v) for v in val)
    return str(val).strip()


def _weighted(val, n: int) -> str:
    s = _to_str(val)
    return " ".join([s] * n) if s else ""


def _theme_search_text(doc: dict) -> str:
    return " ".join(filter(None, [
        _weighted(doc.get("name"), 4),
        _weighted(doc.get("aliases"), 3),
        _weighted(doc.get("description"), 2),
        _weighted(doc.get("examples"), 1),
        _weighted(doc.get("notes"), 1),
    ]))


def _story_search_text(doc: dict) -> str:
    return " ".join(filter(None, [
        _weighted(doc.get("title"), 4),
        _weighted(doc.get("description"), 2),
        _weighted(doc.get("date"), 2),
        _weighted(doc.get("authors"), 2),
    ]))


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _embed_pieces(doc: dict) -> list[str]:
    """Short text pieces to embed: full name/alias phrases + individual content words."""
    pieces = []
    seen: set[str] = set()

    def add(text):
        t = _to_str(text).strip()
        if not t or t.lower() in seen:
            return
        seen.add(t.lower())
        pieces.append(t)
        for w in _content_words(t):
            if w not in seen:
                seen.add(w)
                pieces.append(w)

    if doc["type"] == "theme":
        add(doc.get("name"))
        aliases = doc.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases] if aliases else []
        for alias in aliases:
            add(alias)
    else:
        add(doc.get("title"))
        add(doc.get("name"))

    return pieces


def fetch_documents(ontology) -> list[dict]:
    log.info("Processing ontology documents...")
    todict = ontology.to_dict()
    docs = []

    for raw in todict.get("themes", []):
        doc = dict(raw)
        doc["type"] = "theme"
        doc["search_text"] = _theme_search_text(doc)
        docs.append(doc)

    for raw in todict.get("stories", []):
        doc = dict(raw)
        doc["type"] = "story"
        doc["search_text"] = _story_search_text(doc)
        docs.append(doc)

    for raw in todict.get("collections", []):
        doc = dict(raw)
        doc["type"] = "collection"
        doc["search_text"] = _story_search_text(doc)
        docs.append(doc)

    log.info("Processed %d documents", len(docs))
    return docs


def build(ontology=None) -> tuple:
    if ontology is None:
        log.info("Fetching latest ontology from totolo...")
        ontology = totolo.remote()

    docs = fetch_documents(ontology)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f)

    log.info("Building tantivy index...")
    idx = tantivy.Index(_schema(), path=str(INDEX_DIR))
    writer = idx.writer()
    for doc in docs:
        writer.add_document(tantivy.Document(
            doc_id=doc["name"],
            search_text=doc["search_text"],
        ))
    writer.commit()

    # Build piece-level embeddings: index short phrases and content words per doc,
    # then at search time max-pool scores across all pieces of a document.
    log.info("Building piece-level semantic embeddings...")
    pieces_texts: list[str] = []
    pieces_ids: list[str] = []
    for doc in docs:
        for piece in _embed_pieces(doc):
            pieces_texts.append(piece)
            pieces_ids.append(doc["name"])

    log.info("Embedding %d text pieces from %d documents...", len(pieces_texts), len(docs))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(pieces_texts, show_progress_bar=True, normalize_embeddings=True)

    with open(PIECES_FILE, "w", encoding="utf-8") as f:
        json.dump(pieces_ids, f)
    np.save(str(EMBEDDINGS_FILE), embeddings)

    log.info("Index built: %d documents, %d pieces", len(docs), len(pieces_texts))
    doc_map = {d["name"]: d for d in docs}
    return idx, embeddings, pieces_ids, doc_map


def load() -> tuple:
    with open(DOCS_FILE, encoding="utf-8") as f:
        docs = json.load(f)
    with open(PIECES_FILE, encoding="utf-8") as f:
        pieces_ids = json.load(f)
    embeddings = np.load(str(EMBEDDINGS_FILE))
    doc_map = {d["name"]: d for d in docs}
    idx = tantivy.Index(_schema(), path=str(INDEX_DIR))
    return idx, embeddings, pieces_ids, doc_map


def ensure(force: bool = False) -> tuple:
    if force or not EMBEDDINGS_FILE.exists():
        return build()
    return load()
