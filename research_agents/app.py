#!/usr/bin/env python3
"""Local web app for the multi-agent academic research system.

Run with `python app.py`, then open http://127.0.0.1:8000. Streams pipeline
progress to the browser over Server-Sent Events.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline import OUTPUT_DIR, stream_research

app = FastAPI(title="Multi-Agent Research System")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/research/stream")
async def research_stream(question: str = Query(..., min_length=1)) -> StreamingResponse:
    async def event_source():
        try:
            async for event in stream_research(question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # surface pipeline errors to the browser instead of dying silently
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/api/reports")
async def list_reports() -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in OUTPUT_DIR.glob("*.md"))


@app.get("/api/reports/{filename}")
async def get_report(filename: str) -> FileResponse:
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/markdown")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
