from mcp.server.fastmcp import FastMCP
from totolo_search.search.engine import SearchEngine

mcp = FastMCP("totolo-search")
_engine: SearchEngine | None = None


def _get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


@mcp.tool()
def search(query: str, limit: int = 20) -> list[dict]:
    """Search themeontology.org themes, stories, and collections by keyword, fuzzy match, and semantic similarity."""
    return _get_engine().search(query, limit)


@mcp.tool()
def get_document(name: str) -> dict | None:
    """Retrieve a full document by its unique name (as returned in search results)."""
    return _get_engine().get_document(name)


def run():
    mcp.run()
