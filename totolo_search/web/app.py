from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from totolo_search.search.engine import SearchEngine

STATIC_DIR = Path(__file__).parent / "static"
_engine: SearchEngine | None = None


def _get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


def create_app() -> FastAPI:
    app = FastAPI(title="totolo-search")

    @app.get("/search")
    def search(q: str, limit: int = 20) -> list[dict]:
        return _get_engine().search(q, limit)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
