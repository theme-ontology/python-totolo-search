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
# The annotation index is SEPARATE and built/loaded on its own, so the main index stays lean and the
# ~58k motivation embeddings only cost memory when someone actually searches annotations.
ANN_INDEX_DIR = DATA_DIR / "ann_index"
ANN_DOCS_FILE = DATA_DIR / "annotations.json"
ANN_EMB_FILE = DATA_DIR / "ann_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

# Three weighted fields (eDisMax-style), matching totolo-search-web. Boosts are applied at query time.
FIELDS = ["search_title", "search_body", "search_misc"]
BOOSTS = {"search_title": 2.5, "search_body": 1.0, "search_misc": 0.4}

WEIGHTS = {"choice themes": "choice", "major themes": "major",
           "minor themes": "minor", "not themes": "not"}


def _schema():
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("doc_id", stored=True)
    for f in FIELDS:
        builder.add_text_field(f, stored=False, tokenizer_name="en_stem")
    return builder.build()


def _to_str(val) -> str:
    if not val:
        return ""
    if isinstance(val, (list, tuple)):
        return " ".join(str(v) for v in val)
    return str(val).strip()


def _join(*vals) -> str:
    return " ".join(s for s in (_to_str(v) for v in vals) if s)


def _doc_search_fields(doc: dict) -> dict:
    """The three weighted search fields for a theme / story / collection (mirrors web corpus.ts)."""
    if doc["type"] == "theme":
        title = _join(doc.get("name"), doc.get("aliases"))
        misc = _join(doc.get("examples"), doc.get("notes"))
    else:  # story / collection
        title = _join(doc.get("name"), doc.get("title"), doc.get("date"), doc.get("authors"))
        misc = _join(doc.get("other keywords"), doc.get("story format"))
    return {"search_title": title, "search_body": _to_str(doc.get("description")), "search_misc": misc}


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "it", "its", "that", "this",
    "was", "are", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "not", "no",
    "so", "then", "when", "where", "which", "who", "what", "how", "if",
    "i", "we", "he", "she", "they", "you", "me", "him", "her", "us", "them",
}


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _embed_pieces(doc: dict) -> list[str]:
    """Short text pieces to embed: full name/alias/title phrases + individual content words."""
    pieces, seen = [], set()

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
    """Themes + stories + collections, each enriched with its three weighted search fields."""
    log.info("Processing ontology documents...")
    todict = ontology.to_dict()
    docs = []
    for typ, key in (("theme", "themes"), ("story", "stories"), ("collection", "collections")):
        for raw in todict.get(key, []):
            doc = dict(raw)
            doc["type"] = typ
            doc.update(_doc_search_fields(doc))
            docs.append(doc)
    log.info("Processed %d documents", len(docs))
    return docs


def fetch_annotations(ontology) -> list[dict]:
    """One document per story-theme annotation that carries a motivation. The motivation is the body;
    the searchable identity is '<theme> <story>'. Built straight from to_dict()'s weighted theme lists."""
    todict = ontology.to_dict()
    anns = []
    for key in ("stories", "collections"):
        for st in todict.get(key, []):
            story = st.get("name", "")
            title = _to_str(st.get("title"))
            for wkey, level in WEIGHTS.items():
                for entry in st.get(wkey, []) or []:
                    theme = entry.get("name", "")
                    motivation = _to_str(entry.get("motivation"))
                    if not (theme and motivation):
                        continue
                    anns.append({
                        "name": f"{story} :: {theme}",
                        "type": "story-theme",
                        "story": story, "story_title": title, "theme": theme,
                        "level": level, "motivation": motivation,
                        "search_title": _join(theme, story, title),
                        "search_body": motivation,
                        "search_misc": level,
                    })
    return anns


def _build_tantivy(docs: list[dict], index_dir: Path):
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    idx = tantivy.Index(_schema(), path=str(index_dir))
    writer = idx.writer()
    for doc in docs:
        writer.add_document(tantivy.Document(
            doc_id=doc["name"],
            search_title=doc["search_title"], search_body=doc["search_body"], search_misc=doc["search_misc"]))
    writer.commit()
    return idx


def _embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)


def build(ontology=None) -> tuple:
    """Build the MAIN index (themes/stories/collections): tantivy + piece-level semantic embeddings."""
    if ontology is None:
        log.info("Fetching latest ontology from totolo...")
        ontology = totolo.remote()
    docs = fetch_documents(ontology)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Building tantivy index (%d docs)...", len(docs))
    idx = _build_tantivy(docs, INDEX_DIR)
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f)

    log.info("Building piece-level semantic embeddings...")
    pieces_texts, pieces_ids = [], []
    for doc in docs:
        for piece in _embed_pieces(doc):
            pieces_texts.append(piece)
            pieces_ids.append(doc["name"])
    log.info("Embedding %d pieces from %d docs...", len(pieces_texts), len(docs))
    embeddings = _embed(pieces_texts)
    with open(PIECES_FILE, "w", encoding="utf-8") as f:
        json.dump(pieces_ids, f)
    np.save(str(EMBEDDINGS_FILE), embeddings)
    log.info("Main index built: %d docs, %d pieces", len(docs), len(pieces_texts))
    return idx, embeddings, pieces_ids, {d["name"]: d for d in docs}


def build_annotations(ontology=None) -> tuple:
    """Build the SEPARATE annotation index: tantivy + ONE motivation embedding per annotation."""
    if ontology is None:
        ontology = totolo.remote()
    anns = fetch_annotations(ontology)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Building annotation tantivy index (%d annotations)...", len(anns))
    idx = _build_tantivy(anns, ANN_INDEX_DIR)
    with open(ANN_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(anns, f)

    log.info("Embedding %d motivations...", len(anns))
    embeddings = _embed([a["motivation"] for a in anns])
    np.save(str(ANN_EMB_FILE), embeddings)
    log.info("Annotation index built: %d annotations", len(anns))
    return idx, embeddings, {a["name"]: a for a in anns}


def load() -> tuple:
    with open(DOCS_FILE, encoding="utf-8") as f:
        docs = json.load(f)
    with open(PIECES_FILE, encoding="utf-8") as f:
        pieces_ids = json.load(f)
    # mmap (file-backed) so these pages are reclaimable under memory pressure instead of pinned anon
    # RAM — the kernel can drop and re-fault them from disk rather than OOM-killing.
    embeddings = np.load(str(EMBEDDINGS_FILE), mmap_mode="r")
    idx = tantivy.Index(_schema(), path=str(INDEX_DIR))
    return idx, embeddings, pieces_ids, {d["name"]: d for d in docs}


def load_annotations() -> tuple:
    with open(ANN_DOCS_FILE, encoding="utf-8") as f:
        anns = json.load(f)
    embeddings = np.load(str(ANN_EMB_FILE), mmap_mode="r")   # mmap: reclaimable (see load())
    idx = tantivy.Index(_schema(), path=str(ANN_INDEX_DIR))
    return idx, embeddings, anns


def ensure(force: bool = False) -> tuple:
    if force or not EMBEDDINGS_FILE.exists():
        return build()
    return load()


def annotations_built() -> bool:
    return ANN_EMB_FILE.exists() and ANN_DOCS_FILE.exists()
