"""MCP server exposing the Theme Ontology (themeontology.org) search API.

Tools, at a glance:
  search(q)              — find anything: themes, stories, collections
  search_themes(q)       — find literary THEMES (e.g. "betrayal", "destiny")
  search_stories(q)      — find STORIES (films, books, episodes…)
  search_collections(q)  — find COLLECTIONS (named groups of stories)
  search_annotations(q)  — find story-theme ANNOTATIONS by their motivation text
                           (WHY a given theme applies to a given story)
  list_themes(offset)    — page through ALL theme names
  list_stories(offset)   — page through ALL story names (title + date)
  get_document(name)     — the full record for a theme/story/collection/annotation

Query syntax (all search_* tools): "exact phrase", +required term, -excluded term; bare words are
optional/boosted. Names returned by one tool are the exact `name` to pass to get_document.

Every search_* tool takes two switches:
  fuzzy=True     allow NEAR (1-edit) typo matches, appended below exact hits at low weight. Set
                 False for exact (stemmed) matching only.
  semantic=True  blend in embedding-based semantic search. Set False for keyword-only (raw BM25)
                 results — useful when you (an LLM) intend to rerank the candidates yourself.
"""
from mcp.server.fastmcp import FastMCP
from totolo_search.search.engine import SearchEngine

mcp = FastMCP("totolo-search")
_engine: SearchEngine | None = None

_SYNTAX = ('Query syntax: "exact phrase", +required, -excluded; plain words optional. '
           'fuzzy=False for exact-only; semantic=False for keyword-only (you rerank).')


def _get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


@mcp.tool()
def search(query: str, limit: int = 20, fuzzy: bool = True, semantic: bool = True) -> list[dict]:
    """Search ALL themeontology.org documents (themes, stories, collections) by keyword + semantic
    similarity. Returns ranked {name, type, description, …}. Use the type-specific tools to narrow.
    """ + _SYNTAX
    return _get_engine().search(query, limit, fuzzy=fuzzy, semantic=semantic)


@mcp.tool()
def search_themes(query: str, limit: int = 20, fuzzy: bool = True, semantic: bool = True) -> list[dict]:
    """Search only literary THEMES (abstract ideas like 'betrayal', 'coming of age', 'destiny').
    Start here to find the canonical theme name + description. """ + _SYNTAX
    return _get_engine().search_themes(query, limit, fuzzy=fuzzy, semantic=semantic)


@mcp.tool()
def search_stories(query: str, limit: int = 20, fuzzy: bool = True, semantic: bool = True) -> list[dict]:
    """Search only STORIES (films, books, plays, episodes). Returns {name, title, date, description}.
    The `name` is the unique id to pass to get_document. """ + _SYNTAX
    return _get_engine().search_stories(query, limit, fuzzy=fuzzy, semantic=semantic)


@mcp.tool()
def search_collections(query: str, limit: int = 20, fuzzy: bool = True, semantic: bool = True) -> list[dict]:
    """Search only COLLECTIONS (named groups of stories, e.g. a series or a curated set). """ + _SYNTAX
    return _get_engine().search_collections(query, limit, fuzzy=fuzzy, semantic=semantic)


@mcp.tool()
def search_annotations(query: str, limit: int = 20, fuzzy: bool = True, semantic: bool = True) -> list[dict]:
    """Search the story-theme ANNOTATIONS by their motivation prose — i.e. find where the ontology
    explains WHY a theme applies to a story. Returns {name, story, story_title, theme, level,
    motivation}; `level` is choice/major/minor/not; `name` is "<story> :: <theme>". Use this to find
    precedent for how a theme has been argued/applied. """ + _SYNTAX
    return _get_engine().search_annotations(query, limit, fuzzy=fuzzy, semantic=semantic)


@mcp.tool()
def list_themes(offset: int = 0, limit: int = 200) -> dict:
    """Page through ALL theme names (sorted). Returns {total, offset, limit, items:[{name}]}.
    For exhaustively going through every theme — call repeatedly, increasing `offset` by `limit`."""
    return _get_engine().list_themes(offset, limit)


@mcp.tool()
def list_stories(offset: int = 0, limit: int = 200) -> dict:
    """Page through ALL story names (sorted). Returns {total, offset, limit, items:[{name,title,date}]}.
    For exhaustively going through every story — call repeatedly, increasing `offset` by `limit`."""
    return _get_engine().list_stories(offset, limit)


@mcp.tool()
def list_collections(offset: int = 0, limit: int = 200) -> dict:
    """Page through ALL collection names (sorted). Returns {total, offset, limit, items:[{name,title}]}."""
    return _get_engine().list_collections(offset, limit)


@mcp.tool()
def get_document(name: str) -> dict | None:
    """Retrieve the FULL record for a theme/story/collection/annotation by its exact `name` (as
    returned by any search/list tool). Themes include description, aliases, parents; stories include
    title, date, authors, and their annotated themes; an annotation (name "<story> :: <theme>")
    returns {story, story_title, theme, level, motivation}."""
    return _get_engine().get_document(name)


def run():
    mcp.run()
