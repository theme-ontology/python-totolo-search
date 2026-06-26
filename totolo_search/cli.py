import argparse
import threading
import time
import webbrowser
import totolo


def _get_ontology(version: str | None):
    if version is not None:
        valid = {v[0] for v in totolo.remote.versions()}
        assert version in valid, f"Unknown version {version!r}. Valid: {sorted(valid)}"
        return totolo.remote.version(version)
    return totolo.remote()


def main():
    parser = argparse.ArgumentParser(description="Search themeontology.org themes and stories")
    parser.add_argument("--mcp-only", action="store_true", help="Start MCP server on stdio (for IDE/agent use)")
    parser.add_argument("--build-index", action="store_true", help="Fetch data and rebuild the search index")
    parser.add_argument("--no-annotations", action="store_true", help="With --build-index: skip the (large) annotation index")
    parser.add_argument("--version", help="Ontology version to index (default: latest)")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    args = parser.parse_args()

    if args.build_index:
        from totolo_search.search.index import build, build_annotations
        ont = _get_ontology(args.version)
        build(ont)                                   # main index first (frees its embeddings before the next step)
        if not args.no_annotations:
            build_annotations(ont)                   # separate, larger annotation index
        return

    if args.mcp_only:
        from totolo_search.mcp.server import run
        run()
        return

    import uvicorn
    from totolo_search.web.app import create_app

    app = create_app()
    url = f"http://{args.host}:{args.port}"

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)
