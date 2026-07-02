"""FastAPI application for AI-powered candidate ranking."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from config import get_settings
from app.models.schemas import RankingResponse
from app.services.pipeline import RankingPipeline


settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    description="Semantic candidate ranking with embeddings, FAISS, hybrid scoring, and Groq reranking.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict[str, Path] = {}


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""

    return {"status": "ok", "environment": settings.environment}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    """Simple upload interface for hackathon demos."""

    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AI Candidate Ranking</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 2rem; color: #18212f; }
          main { max-width: 860px; margin: auto; }
          section { margin: 1rem 0; padding: 1rem 0; border-bottom: 1px solid #d7dce5; }
          button { padding: .7rem 1rem; border: 0; background: #155eef; color: white; cursor: pointer; }
          input { margin: .4rem 0; }
          table { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: .9rem; }
          th, td { border: 1px solid #d7dce5; padding: .55rem; text-align: left; }
        </style>
      </head>
      <body>
        <main>
          <h1>AI Candidate Ranking</h1>
          <section>
            <h2>Upload Job</h2>
            <input id="job" type="file" accept=".txt,.pdf,.docx" />
            <button onclick="upload('/upload-job','job')">Upload</button>
          </section>
          <section>
            <h2>Upload Candidates</h2>
            <input id="candidates" type="file" accept=".csv,.json" />
            <button onclick="upload('/upload-candidates','candidates')">Upload</button>
          </section>
          <section>
            <button onclick="rank()">Rank Candidates</button>
            <a href="/results" target="_blank">Download CSV</a>
            <div id="status"></div>
            <div id="results"></div>
          </section>
        </main>
        <script>
          async function upload(url, inputId) {
            const file = document.getElementById(inputId).files[0];
            if (!file) return;
            const form = new FormData();
            form.append('file', file);
            const response = await fetch(url, { method: 'POST', body: form });
            document.getElementById('status').textContent = JSON.stringify(await response.json());
          }
          async function rank() {
            document.getElementById('status').textContent = 'Ranking...';
            const response = await fetch('/rank', { method: 'POST' });
            const payload = await response.json();
            document.getElementById('status').textContent = payload.output_file || payload.detail;
            const rows = (payload.results || []).slice(0, 10).map(item => `
              <tr><td>${item.rank}</td><td>${item.candidate_name}</td><td>${item.overall_score}</td><td>${item.reason}</td></tr>
            `).join('');
            document.getElementById('results').innerHTML = rows ? `<table><thead><tr><th>Rank</th><th>Name</th><th>Score</th><th>Reason</th></tr></thead><tbody>${rows}</tbody></table>` : '';
          }
        </script>
      </body>
    </html>
    """


@app.post("/upload-job")
async def upload_job(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload a job description file."""

    path = await _save_upload(file, {"txt", "pdf", "docx"})
    _state["job_path"] = path
    return {"message": "job uploaded", "path": str(path)}


@app.post("/upload-candidates")
async def upload_candidates(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload a candidate CSV or JSON file."""

    path = await _save_upload(file, {"csv", "json"})
    _state["candidates_path"] = path
    return {"message": "candidates uploaded", "path": str(path)}


@app.post("/rank", response_model=RankingResponse)
def rank() -> RankingResponse:
    """Rank uploaded candidates against the uploaded job description."""

    job_path = _state.get("job_path")
    candidates_path = _state.get("candidates_path")
    if not job_path or not candidates_path:
        raise HTTPException(status_code=400, detail="Upload a job file and candidate file first.")
    try:
        results, output_path = RankingPipeline(settings).rank_from_files(job_path, candidates_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RankingResponse(results=results, output_file=str(output_path))


@app.get("/results", response_model=None)
def results():
    """Download the latest ranked candidate CSV."""

    output_path = settings.output_dir / "ranked_candidates.csv"
    if not output_path.exists():
        return {"message": "No results available yet. Run /rank first."}
    return FileResponse(output_path, media_type="text/csv", filename="ranked_candidates.csv")


@app.get("/results/preview")
def results_preview() -> list[dict[str, object]]:
    """Preview the latest ranking output."""

    output_path = settings.output_dir / "ranked_candidates.csv"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="No results available yet.")
    return pd.read_csv(output_path).to_dict(orient="records")


async def _save_upload(file: UploadFile, allowed_extensions: set[str]) -> Path:
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}",
        )
    target = settings.upload_dir / Path(file.filename or f"upload.{suffix}").name
    content = await file.read()
    target.write_bytes(content)
    return target
