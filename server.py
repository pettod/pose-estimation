"""Local web dashboard: stats from stats.json + annotated video from output/."""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Factory floor dashboard")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/stats")
async def api_stats():
    if not config.STATS_JSON_PATH.is_file():
        return JSONResponse(
            {
                "error": "stats.json not found. Run `python main.py` on a video first.",
            },
            status_code=404,
        )
    data = json.loads(config.STATS_JSON_PATH.read_text(encoding="utf-8"))
    return data


@app.get("/media/output/{name:path}")
async def serve_output_video(name: str):
    """Serve files only from OUTPUT_DIR (basename only, no traversal)."""
    safe = Path(name).name
    if safe != name or ".." in name:
        raise HTTPException(status_code=404)
    path = (config.OUTPUT_DIR / safe).resolve()
    out = config.OUTPUT_DIR.resolve()
    try:
        path.relative_to(out)
    except ValueError:
        raise HTTPException(status_code=404)
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="video/mp4")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def main():
    import uvicorn

    print(f"Open http://127.0.0.1:5000  (project root: {config.PROJECT_ROOT})")
    uvicorn.run(app, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
