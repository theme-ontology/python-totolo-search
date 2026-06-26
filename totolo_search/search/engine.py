import logging
import re

import numpy as np

from . import index as idx_module

log = logging.getLogger(__name__)

_RRF_K = 60
_SEM_THRESHOLD = 0.50        # main (piece max-pool) — preserves prior behaviour
_SEM_THRESHOLD_ANN = 0.40    # annotation motivations are longer/descriptive

# --- query parsing: a port of totolo-search-web's parseOperators (eDisMax-style) -----------------
# Supports "exact phrases", +required, -excluded, and free terms.
_OP_RE = re.compile(r'^([+-])("[^"]+"|\S+)')
_QUOTED_RE = re.compile(r'^"([^"]+)"')


def parse_operators(text: str) -> dict:
    free, phrases, required, excluded = [], [], [], []
    s = (text or "").strip()
    while s:
        s = s.lstrip()
        if not s:
            break
        m = _OP_RE.match(s)
        if m:
            op, rest = m.group(1), m.group(2)
            s = s[m.end():]
            qm = _QUOTED_RE.match(rest)
            term = qm.group(1) if qm else rest
            if op == "+":
                required.append(term)
                (phrases if qm else free).append(term)
            else:
                excluded.append(term)
            continue
        qm = _QUOTED_RE.match(s)
        if qm:
            phrases.append(qm.group(1))
            free.append(qm.group(1))
            s = s[qm.end():]
            continue
        wm = re.match(r"^\S+", s)
        if wm:
            free.append(wm.group(0))
            s = s[wm.end():]
            continue
        break
    free_text = " ".join(free)
    if not free_text and required:
        free_text = " ".join(required)
    return {"free_text": free_text, "phrases": phrases, "required": required, "excluded": excluded}


def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _to_tantivy_query(parsed: dict) -> str:
    """Rebuild a clean tantivy query string from parsed operators. Only alphanumeric tokens reach the
    parser, so user punctuation can't break it. + => MUST, - => MUST NOT, phrases => "...", rest SHOULD."""
    must, mustnot, should = [], [], []
    req_words = set()
    for t in parsed["required"]:
        ts = _toks(t)
        if not ts:
            continue
        if len(ts) > 1:
            must.append('"%s"' % " ".join(ts))
        else:
            must.append(ts[0])
            req_words.add(ts[0])
    for t in parsed["excluded"]:
        ts = _toks(t)
        if not ts:
            continue
        mustnot.append('"%s"' % " ".join(ts) if len(ts) > 1 else ts[0])
    phrase_words = set()
    for p in parsed["phrases"]:
        ts = _toks(p)
        phrase_words.update(ts)
        if p not in parsed["required"] and ts:
            should.append('"%s"' % " ".join(ts))
    for w in _toks(parsed["free_text"]):
        if w not in req_words and w not in phrase_words:
            should.append(w)
    return " ".join(["+" + m for m in must] + ["-" + m for m in mustnot] + should)


class SearchEngine:
    def __init__(self):
        self._index, self._embeddings, self._piece_ids, self._doc_map = idx_module.ensure()
        self._model = None
        self._ann = None   # lazily loaded: (index, embeddings, names, name_map)
        self._by_type: dict[str, list[str]] = {}
        for name, d in self._doc_map.items():
            self._by_type.setdefault(d["type"], []).append(name)
        for v in self._by_type.values():
            v.sort()

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(idx_module.MODEL_NAME)
        return self._model

    # ---- lexical (tantivy, multi-field + boosts) ----
    def _lexical(self, index, parsed: dict, k: int, fuzzy: bool = True) -> list[str]:
        """Exact (stemmed) matches first, at full weight. When `fuzzy`, NEAR matches (Levenshtein
        distance 1, no prefix expansion) are appended AFTER the exact ones — so fuzzy only fills gaps
        and never outranks an exact hit (matching totolo-search-web's low fuzzy weighting)."""
        qstr = _to_tantivy_query(parsed)
        if not qstr.strip():
            return []
        searcher = index.searcher()

        def run(fuzzy_fields):
            try:
                q = index.parse_query(qstr, idx_module.FIELDS, field_boosts=idx_module.BOOSTS,
                                      fuzzy_fields=fuzzy_fields)
            except Exception:
                return []
            return [searcher.doc(addr)["doc_id"][0] for _s, addr in searcher.search(q, k).hits]

        exact = run({})
        if not fuzzy or len(exact) >= k:
            return exact[:k]
        seen = set(exact)
        near = [d for d in run({f: (False, 1, True) for f in idx_module.FIELDS}) if d not in seen]
        return (exact + near)[:k]

    # ---- semantic ----
    def _semantic_main(self, query: str, k: int) -> list[str]:
        q = self._get_model().encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ q
        best: dict[str, float] = {}
        for i, name in enumerate(self._piece_ids):
            s = float(scores[i])
            if s > best.get(name, -1.0):
                best[name] = s
        filt = {n: v for n, v in best.items() if v >= _SEM_THRESHOLD}
        return sorted(filt, key=filt.get, reverse=True)[:k]

    def _semantic_ann(self, emb, names, query: str, k: int) -> list[str]:
        q = self._get_model().encode([query], normalize_embeddings=True)[0]
        scores = emb @ q
        order = np.argsort(-scores)[: k * 2]
        return [names[i] for i in order if scores[i] >= _SEM_THRESHOLD_ANN][:k]

    @staticmethod
    def _rrf(rankings, weights):
        scores: dict[str, float] = {}
        for ranking, w in zip(rankings, weights):
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + w / (_RRF_K + rank + 1)
        return scores

    def search(self, query: str, limit: int = 20, types=None,
               fuzzy: bool = True, semantic: bool = True) -> list[dict]:
        """Search themes/stories/collections, optional `types` filter.
        fuzzy=False -> exact (stemmed) matches only. semantic=False -> keyword-only (no embeddings),
        returning the raw BM25 ranking for the caller (e.g. an LLM) to rerank itself."""
        parsed = parse_operators(query)
        pool = None
        if types:
            types = {types} if isinstance(types, str) else set(types)
            pool = {n for n, d in self._doc_map.items() if d["type"] in types}
        k = limit if pool is None else max(200, limit * 10)
        kw = self._lexical(self._index, parsed, k, fuzzy=fuzzy)
        sem = self._semantic_main(parsed["free_text"] or query, k) if semantic else []
        if pool is not None:
            kw = [n for n in kw if n in pool]
            sem = [n for n in sem if n in pool]
        kw_w, sem_w = 2.0, 1.0
        kw_s = {d: kw_w / (_RRF_K + r + 1) for r, d in enumerate(kw)}
        sem_s = {d: sem_w / (_RRF_K + r + 1) for r, d in enumerate(sem)}
        if sem:
            order = sorted(self._rrf([kw, sem], [kw_w, sem_w]),
                           key=lambda x: kw_s.get(x, 0) + sem_s.get(x, 0), reverse=True)
        else:
            order = kw   # keyword-only: leave reranking to the caller
        out = []
        for name in order[:limit]:
            doc = self._doc_map.get(name)
            if not doc:
                continue
            desc = idx_module._to_str(doc.get("description"))
            entry = {"name": name, "type": doc["type"],
                     "description": desc[:160] + ("..." if len(desc) > 160 else ""),
                     "kw_score": round(kw_s.get(name, 0.0), 4), "sem_score": round(sem_s.get(name, 0.0), 4)}
            if doc["type"] in ("story", "collection"):
                entry["title"] = idx_module._to_str(doc.get("title"))
                entry["date"] = idx_module._to_str(doc.get("date"))
            out.append(entry)
        return out

    def search_themes(self, query, limit=20, fuzzy=True, semantic=True):
        return self.search(query, limit, types=["theme"], fuzzy=fuzzy, semantic=semantic)

    def search_stories(self, query, limit=20, fuzzy=True, semantic=True):
        return self.search(query, limit, types=["story"], fuzzy=fuzzy, semantic=semantic)

    def search_collections(self, query, limit=20, fuzzy=True, semantic=True):
        return self.search(query, limit, types=["collection"], fuzzy=fuzzy, semantic=semantic)

    # ---- annotations (separate, lazily loaded) ----
    def _ensure_ann(self):
        if self._ann is None:
            idx, emb, anns = idx_module.load_annotations()
            self._ann = (idx, emb, [a["name"] for a in anns], {a["name"]: a for a in anns})
        return self._ann

    def search_annotations(self, query, limit=20, fuzzy=True, semantic=True):
        """Search the motivation prose of story-theme annotations (WHY a theme applies to a story).
        Semantic leads when on (motivations are prose); semantic=False gives keyword-only."""
        if not idx_module.annotations_built():
            return []
        idx, emb, names, name_map = self._ensure_ann()
        parsed = parse_operators(query)
        kw = self._lexical(idx, parsed, limit * 3, fuzzy=fuzzy)
        sem = self._semantic_ann(emb, names, parsed["free_text"] or query, limit * 3) if semantic else []
        if sem:
            order = [n for n, _ in sorted(self._rrf([sem, kw], [2.0, 1.0]).items(),
                                          key=lambda kv: kv[1], reverse=True)]
        else:
            order = kw
        out = []
        for name in order[:limit]:
            a = name_map.get(name)
            if not a:
                continue
            mot = a["motivation"]
            out.append({"name": name, "type": "story-theme",
                        "story": a["story"], "story_title": a.get("story_title", ""),
                        "theme": a["theme"], "level": a["level"],
                        "motivation": mot[:300] + ("..." if len(mot) > 300 else "")})
        return out

    # ---- enumeration (for "go through every story/theme" workflows) ----
    def _page(self, names, offset, limit, fields=None):
        page = names[offset:offset + limit]
        items = []
        for n in page:
            it = {"name": n}
            for f in (fields or []):
                it[f] = idx_module._to_str(self._doc_map[n].get(f))
            items.append(it)
        return {"total": len(names), "offset": offset, "limit": limit, "items": items}

    def list_themes(self, offset: int = 0, limit: int = 200) -> dict:
        return self._page(self._by_type.get("theme", []), offset, limit)

    def list_stories(self, offset: int = 0, limit: int = 200) -> dict:
        return self._page(self._by_type.get("story", []), offset, limit, ["title", "date"])

    def list_collections(self, offset: int = 0, limit: int = 200) -> dict:
        return self._page(self._by_type.get("collection", []), offset, limit, ["title"])

    def get_document(self, name: str) -> dict | None:
        d = self._doc_map.get(name)
        if d:
            return d
        if idx_module.annotations_built() and " :: " in (name or ""):
            _, _, _, name_map = self._ensure_ann()
            a = name_map.get(name)
            if a:   # drop the internal index fields; return the meaningful record
                return {k: v for k, v in a.items()
                        if k not in ("search_title", "search_body", "search_misc")}
        return None
