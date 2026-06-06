import logging
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein as _lev
from . import index as idx_module

log = logging.getLogger(__name__)

_RRF_K = 60
_SEM_THRESHOLD = 0.50


def _rrf(rankings: list[list[str]], weights: list[float]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


class SearchEngine:
    def __init__(self):
        self._index, self._embeddings, self._piece_ids, self._doc_map = idx_module.ensure()
        self._model = None
        self._vocab = self._build_vocab()

    def _build_vocab(self) -> list[str]:
        words: set[str] = set()
        for doc in self._doc_map.values():
            for field in ("name", "title"):
                val = doc.get(field) or ""
                words.update(val.lower().split())
        return list(words)

    def _expand_query(self, query: str) -> str:
        terms = query.lower().split()
        expanded = []
        for term in terms:
            if len(term) <= 3:
                expanded.append(term)
                continue
            # Dynamic cutoff: only correct single-char typos (edit_distance == 1)
            cutoff = 1.0 - 1.0 / len(term)
            match = process.extractOne(term, self._vocab, scorer=_lev.normalized_similarity, score_cutoff=cutoff)
            if match and match[0] != term:
                log.debug("Expanded %r -> %r", term, match[0])
                expanded.append(match[0])
            else:
                expanded.append(term)
        return " ".join(expanded)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(idx_module.MODEL_NAME)
        return self._model

    def _keyword_search(self, query: str, limit: int) -> list[str]:
        try:
            q = self._index.parse_query(query, ["search_text"])
        except Exception:
            return []
        searcher = self._index.searcher()
        results = searcher.search(q, limit)
        return [searcher.doc(addr)["doc_id"][0] for _score, addr in results.hits]

    def _semantic_search(self, query: str, limit: int) -> list[str]:
        q_emb = self._get_model().encode([query], normalize_embeddings=True)[0]
        # cosine similarity against all piece embeddings (already unit-normalized)
        piece_scores = self._embeddings @ q_emb
        # max-pool per document: best matching piece wins
        doc_best: dict[str, float] = {}
        for i, doc_name in enumerate(self._piece_ids):
            s = float(piece_scores[i])
            if s > doc_best.get(doc_name, -1.0):
                doc_best[doc_name] = s
        # filter below threshold to avoid polluting with irrelevant results
        filtered = {k: v for k, v in doc_best.items() if v >= _SEM_THRESHOLD}
        return sorted(filtered, key=filtered.get, reverse=True)[:limit]

    def search(self, query: str, limit: int = 20, kw_weight: float = 2.0, sem_weight: float = 1.0) -> list[dict]:
        expanded = self._expand_query(query)
        kw_ids = self._keyword_search(expanded, limit)
        sem_ids = self._semantic_search(expanded, limit)

        kw_scores = {doc_id: kw_weight / (_RRF_K + rank + 1) for rank, doc_id in enumerate(kw_ids)}
        sem_scores = {doc_id: sem_weight / (_RRF_K + rank + 1) for rank, doc_id in enumerate(sem_ids)}

        merged = _rrf([kw_ids, sem_ids], [kw_weight, sem_weight])[:limit]

        results = []
        for name in merged:
            doc = self._doc_map.get(name)
            if not doc:
                continue
            desc = doc.get("description") or ""
            entry: dict = {
                "name": name,
                "type": doc["type"],
                "description": desc[:100] + ("..." if len(desc) > 100 else ""),
                "kw_score": round(kw_scores.get(name, 0.0), 4),
                "sem_score": round(sem_scores.get(name, 0.0), 4),
            }
            if doc["type"] in ("story", "collection"):
                entry["title"] = doc.get("title") or ""
            results.append(entry)
        return results

    def get_document(self, name: str) -> dict | None:
        return self._doc_map.get(name)
