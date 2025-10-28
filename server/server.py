#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
# JSON tracks are still produced under render/output
OUTPUT_DIR = ROOT.parent / "render" / "output"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web app (index.html and assets)
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")


@app.get("/api/scenes")
def list_scenes():
    if not OUTPUT_DIR.exists():
        return []
    return sorted([p.stem for p in OUTPUT_DIR.glob("*.json")])


@app.get("/api/scene/{name}")
def get_scene(name: str):
    fp = OUTPUT_DIR / f"{name}.json"
    if not fp.exists():
        raise HTTPException(404, detail="Not found")
    try:
        return json.loads(fp.read_text())
    except Exception:
        raise HTTPException(500, detail="Failed to read scene JSON")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


