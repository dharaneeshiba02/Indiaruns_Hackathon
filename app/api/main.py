"""FastAPI application for AI-powered candidate ranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import get_settings
from app.models.schemas import RankingResponse
from app.services.pipeline import RankingPipeline
from app.services.gemini_client import GeminiRecruiterClient


settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    description="Semantic candidate ranking with embeddings, FAISS, hybrid scoring, and Gemini reranking.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict[str, Any] = {
    "job_path": None,
    "candidates_paths": [],
    "job": None,
    "results": None,
}

class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "environment": settings.environment}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    """Interactive upload interface for hackathon demos with premium UI."""
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AI Candidate Ranking</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
          :root {
            color-scheme: dark;
            --bg-color: #0d1117;
            --surface-color: rgba(22, 27, 34, 0.7);
            --border-color: rgba(48, 54, 61, 0.8);
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-hover: #3182ce;
            --success: #2ea043;
            --danger: #f85149;
            --glass-bg: rgba(22, 27, 34, 0.65);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            background: radial-gradient(circle at 10% 20%, rgb(14, 18, 25) 0%, rgb(11, 15, 20) 90%);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            padding-bottom: 60px;
          }
          /* Custom scrollbar */
          ::-webkit-scrollbar { width: 8px; height: 8px; }
          ::-webkit-scrollbar-track { background: var(--bg-color); }
          ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
          ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

          main {
            width: min(1200px, calc(100% - 40px));
            margin: 0 auto;
            padding: 40px 0;
            display: flex;
            flex-direction: column;
            gap: 24px;
          }
          header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px 32px;
            box-shadow: var(--glass-shadow);
          }
          .title-area h1 {
            margin: 0;
            font-size: 32px;
            font-weight: 600;
            background: -webkit-linear-gradient(45deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
          }
          .title-area p { margin: 8px 0 0; color: var(--text-secondary); font-size: 15px; }
          
          .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 500;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 999px;
            border: 1px solid var(--glass-border);
          }
          .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
            transition: background 0.3s ease;
          }
          .status-dot.ready { background: var(--success); box-shadow: 0 0 8px var(--success); }
          .status-dot.busy { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 1.5s infinite; }
          .status-dot.error { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
          
          @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.7; }
            50% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.7; }
          }

          .layout-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
          }
          @media (min-width: 900px) {
            .layout-grid { grid-template-columns: 380px 1fr; }
          }

          .panel {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--glass-shadow);
            display: flex;
            flex-direction: column;
            gap: 20px;
          }

          .dropzone {
            display: flex;
            flex-direction: column;
            gap: 16px;
            border: 2px dashed rgba(88, 166, 255, 0.3);
            border-radius: 12px;
            background: rgba(13, 17, 23, 0.5);
            padding: 24px;
            transition: all 0.2s ease;
            position: relative;
          }
          .dropzone:hover { border-color: rgba(88, 166, 255, 0.6); background: rgba(88, 166, 255, 0.05); }
          .dropzone.dragover { border-color: var(--accent); background: rgba(88, 166, 255, 0.1); transform: scale(1.02); }
          .dropzone[data-ready="true"] { border-color: var(--success); background: rgba(46, 160, 67, 0.05); }
          
          .dropzone-header { display: flex; justify-content: space-between; align-items: flex-start; }
          .dropzone-title { font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 4px; }
          .dropzone-subtitle { font-size: 13px; color: var(--text-secondary); }
          
          .file-list {
            font-size: 13px;
            color: var(--accent);
            max-height: 80px;
            overflow-y: auto;
            word-break: break-all;
            padding-right: 4px;
          }

          .btn-group { display: flex; gap: 10px; margin-top: auto; }
          button, .file-button {
            border: none;
            border-radius: 8px;
            padding: 10px 16px;
            font-family: inherit;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
          }
          .btn-primary { background: var(--accent); color: #fff; }
          .btn-primary:hover:not(:disabled) { background: #4096ed; box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3); transform: translateY(-1px); }
          .btn-primary:disabled { background: var(--border-color); color: var(--text-secondary); cursor: not-allowed; }
          
          .btn-secondary, .file-button {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
          }
          .btn-secondary:hover, .file-button:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--text-secondary);
          }
          
          input[type="file"] { display: none; }

          .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
          }
          .results-title h2 { margin: 0 0 4px 0; font-size: 20px; font-weight: 600; color: #fff; }
          .results-status { font-size: 14px; color: var(--text-secondary); }

          .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 16px;
          }
          .metric-card {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
          }
          .metric-value { font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 4px; }
          .metric-label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }

          .table-container {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.2);
            overflow-x: auto;
          }
          table { width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }
          th {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 14px 16px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
          }
          td {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: top;
          }
          tr:last-child td { border-bottom: none; }
          tr:hover td { background: rgba(255, 255, 255, 0.02); }

          .score-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            margin-top: 8px;
            overflow: hidden;
          }
          .score-fill {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, #a371f7, #58a6ff);
          }
          
          .empty-state {
            padding: 60px 20px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 15px;
          }

          /* Chatbot UI */
          .chat-panel {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 0;
            box-shadow: var(--glass-shadow);
            display: none; /* Hidden until results exist */
            flex-direction: column;
            height: 500px;
            overflow: hidden;
          }
          .chat-panel.active { display: flex; }
          
          .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.02);
            font-weight: 600;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
          }
          .chat-messages {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
          }
          .msg {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
          }
          .msg.bot {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
          }
          .msg.user {
            background: var(--accent);
            color: #fff;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
          }
          .chat-input-area {
            padding: 16px 24px;
            border-top: 1px solid var(--border-color);
            background: rgba(0, 0, 0, 0.2);
            display: flex;
            gap: 12px;
          }
          .chat-input {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 16px;
            color: #fff;
            font-family: inherit;
            font-size: 14px;
            transition: border-color 0.2s;
          }
          .chat-input:focus { outline: none; border-color: var(--accent); }
          .btn-send {
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 0 20px;
            font-weight: 600;
            cursor: pointer;
          }
          .btn-send:hover { background: #4096ed; }
        </style>
      </head>
      <body>
        <main>
          <header>
            <div class="title-area">
              <h1>AI Candidate Ranking</h1>
              <p>Upload job descriptions and multiple resumes to find the perfect match.</p>
            </div>
            <div class="status-indicator">
              <div class="status-dot ready" id="serverDot"></div>
              <span id="serverState">Ready</span>
            </div>
          </header>

          <div class="layout-grid">
            <!-- Upload Panel -->
            <div class="panel">
              <!-- Job Upload -->
              <div class="dropzone" id="jobZone" data-kind="job" data-ready="false">
                <div class="dropzone-header">
                  <div>
                    <div class="dropzone-title">Job Description</div>
                    <div class="dropzone-subtitle">PDF, DOCX, TXT</div>
                  </div>
                  <div class="status-indicator" style="padding: 4px 10px; font-size: 12px; background:transparent">
                    <span id="jobBadge">Waiting</span>
                  </div>
                </div>
                <div class="file-list" id="jobName"></div>
                <div class="btn-group">
                  <label class="file-button" for="jobInput">Select File</label>
                  <input id="jobInput" type="file" accept=".txt,.pdf,.docx" />
                  <button class="btn-secondary" type="button" onclick="uploadFromInput('job')">Upload</button>
                </div>
              </div>

              <!-- Candidates Upload -->
              <div class="dropzone" id="candidatesZone" data-kind="candidates" data-ready="false">
                <div class="dropzone-header">
                  <div>
                    <div class="dropzone-title">Candidates (Multiple)</div>
                    <div class="dropzone-subtitle">CSV, JSON, PDF, DOCX, TXT</div>
                  </div>
                  <div class="status-indicator" style="padding: 4px 10px; font-size: 12px; background:transparent">
                    <span id="candidatesBadge">Waiting</span>
                  </div>
                </div>
                <div class="file-list" id="candidatesName"></div>
                <div class="btn-group">
                  <label class="file-button" for="candidatesInput">Select Files</label>
                  <input id="candidatesInput" type="file" multiple accept=".csv,.json,.pdf,.docx,.txt" />
                  <button class="btn-secondary" type="button" onclick="uploadFromInput('candidates')">Upload</button>
                </div>
              </div>
            </div>

            <!-- Results Panel -->
            <div class="panel">
              <div class="results-header">
                <div class="results-title">
                  <h2>Ranking Results</h2>
                  <div class="results-status" id="statusText">Upload files and click Rank to begin.</div>
                </div>
                <div class="btn-group">
                  <a class="btn-secondary" href="/results" target="_blank" style="text-decoration:none;">Download CSV</a>
                  <button class="btn-primary" id="rankButton" type="button" onclick="rank()" disabled>Rank Candidates</button>
                </div>
              </div>

              <div class="metrics">
                <div class="metric-card">
                  <div class="metric-value" id="totalMetric">0</div>
                  <div class="metric-label">Ranked</div>
                </div>
                <div class="metric-card">
                  <div class="metric-value" id="topMetric">--</div>
                  <div class="metric-label">Top Score</div>
                </div>
                <div class="metric-card">
                  <div class="metric-value" id="confidenceMetric">--</div>
                  <div class="metric-label">Confidence</div>
                </div>
              </div>

              <div class="table-container" id="resultsContainer">
                <div class="empty-state">Results will appear here after ranking.</div>
              </div>
            </div>
          </div>
          
          <!-- Chat Panel -->
          <div class="chat-panel" id="chatPanel">
            <div class="chat-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
              Recruiter AI Assistant
            </div>
            <div class="chat-messages" id="chatMessages">
              <div class="msg bot">Hello! I've analyzed the ranking results. Feel free to ask me any questions about the candidates, scores, or why someone was ranked highly.</div>
            </div>
            <div class="chat-input-area">
              <input type="text" class="chat-input" id="chatInput" placeholder="Ask about the results..." onkeypress="handleChatKey(event)">
              <button class="btn-send" onclick="sendChat()">Send</button>
            </div>
          </div>
        </main>

        <script>
          const state = { jobUploaded: false, candidatesUploaded: false };

          function updateServerStatus(msg, status) {
            document.getElementById('statusText').textContent = msg;
            const dot = document.getElementById('serverDot');
            const txt = document.getElementById('serverState');
            dot.className = 'status-dot ' + status;
            txt.textContent = status === 'busy' ? 'Working' : status === 'error' ? 'Error' : 'Ready';
          }

          function checkRankReady() {
            document.getElementById('rankButton').disabled = !(state.jobUploaded && state.candidatesUploaded);
          }

          async function uploadFiles(files, kind) {
            if (!files || files.length === 0) return;
            
            const formData = new FormData();
            let endpoint = '';
            
            if (kind === 'job') {
                formData.append('file', files[0]); // Only one job
                endpoint = '/upload-job';
            } else {
                for (let i = 0; i < files.length; i++) {
                    formData.append('files', files[i]);
                }
                endpoint = '/upload-candidates';
            }

            updateServerStatus(`Uploading ${kind}...`, 'busy');
            
            try {
                const response = await fetch(endpoint, { method: 'POST', body: formData });
                if (!response.ok) throw new Error(await response.text());
                
                // Update UI for success
                document.getElementById(`${kind}Zone`).dataset.ready = 'true';
                document.getElementById(`${kind}Badge`).textContent = 'Uploaded';
                
                const names = Array.from(files).map(f => f.name).join(', ');
                document.getElementById(`${kind}Name`).textContent = names;
                
                if (kind === 'job') state.jobUploaded = true;
                if (kind === 'candidates') state.candidatesUploaded = true;
                
                updateServerStatus(`${kind} uploaded successfully.`, 'ready');
                checkRankReady();
            } catch (err) {
                updateServerStatus(`Upload failed: ${err.message}`, 'error');
            }
          }

          async function uploadFromInput(kind) {
            const input = document.getElementById(`${kind}Input`);
            await uploadFiles(input.files, kind);
          }

          async function rank() {
            updateServerStatus('Analyzing and ranking candidates...', 'busy');
            document.getElementById('rankButton').disabled = true;
            
            try {
                const response = await fetch('/rank', { method: 'POST' });
                if (!response.ok) throw new Error(await response.text());
                
                const data = await response.json();
                renderResults(data.results);
                updateServerStatus('Ranking complete.', 'ready');
                
                // Show chatbot
                document.getElementById('chatPanel').classList.add('active');
            } catch (err) {
                updateServerStatus(`Ranking failed: ${err.message}`, 'error');
            } finally {
                checkRankReady();
            }
          }

          function renderResults(results) {
            document.getElementById('totalMetric').textContent = results.length;
            document.getElementById('topMetric').textContent = results.length ? results[0].overall_score.toFixed(1) : '--';
            document.getElementById('confidenceMetric').textContent = results.length ? results[0].confidence_score.toFixed(1) : '--';
            
            if (!results || results.length === 0) {
                document.getElementById('resultsContainer').innerHTML = '<div class="empty-state">No valid candidates found.</div>';
                return;
            }

            let html = `<table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Candidate</th>
                        <th>Overall Score</th>
                        <th>Match Breakdown</th>
                        <th>Detailed Recruiter Note</th>
                    </tr>
                </thead>
                <tbody>`;
                
            results.forEach(r => {
                html += `
                <tr>
                    <td style="font-size: 18px; font-weight:700; color:var(--accent)">#${r.rank}</td>
                    <td>
                        <div style="font-weight:600; color:#fff">${r.candidate_name}</div>
                        <div style="font-size:12px; color:var(--text-secondary); margin-top:4px">${r.candidate_id}</div>
                    </td>
                    <td>
                        <div style="font-size:20px; font-weight:700; color:#fff">${r.overall_score.toFixed(1)}</div>
                        <div class="score-bar"><div class="score-fill" style="width: ${Math.min(100, Math.max(0, r.overall_score))}%"></div></div>
                    </td>
                    <td style="font-size:12px; line-height:1.6">
                        Semantic: <span style="color:#fff">${r.semantic_score.toFixed(1)}</span> &nbsp;
                        Skills: <span style="color:#fff">${r.skill_score.toFixed(1)}</span><br>
                        Exp: <span style="color:#fff">${r.experience_score.toFixed(1)}</span> &nbsp;
                        Edu: <span style="color:#fff">${r.education_score.toFixed(1)}</span>
                    </td>
                    <td style="font-size:13px; color:var(--text-secondary)">${r.reason}</td>
                </tr>`;
            });
            
            html += `</tbody></table>`;
            document.getElementById('resultsContainer').innerHTML = html;
          }

          // Chatbot logic
          function addMessage(text, sender) {
              const chat = document.getElementById('chatMessages');
              const msg = document.createElement('div');
              msg.className = `msg ${sender}`;
              msg.textContent = text;
              chat.appendChild(msg);
              chat.scrollTop = chat.scrollHeight;
          }

          function handleChatKey(e) {
              if (e.key === 'Enter') sendChat();
          }

          async function sendChat() {
              const input = document.getElementById('chatInput');
              const text = input.value.trim();
              if (!text) return;
              
              addMessage(text, 'user');
              input.value = '';
              input.disabled = true;
              
              // Add a temporary loading message
              const chat = document.getElementById('chatMessages');
              const loadingMsg = document.createElement('div');
              loadingMsg.className = 'msg bot';
              loadingMsg.id = 'loadingMsg';
              loadingMsg.innerHTML = '<span style="opacity:0.5">Thinking...</span>';
              chat.appendChild(loadingMsg);
              chat.scrollTop = chat.scrollHeight;

              try {
                  const res = await fetch('/chat', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ question: text })
                  });
                  const data = await res.json();
                  
                  // Remove loading message
                  document.getElementById('loadingMsg').remove();
                  
                  if (res.ok) {
                      addMessage(data.answer, 'bot');
                  } else {
                      addMessage("Sorry, an error occurred.", 'bot');
                  }
              } catch (e) {
                  document.getElementById('loadingMsg').remove();
                  addMessage("Connection error.", 'bot');
              } finally {
                  input.disabled = false;
                  input.focus();
              }
          }

          // Drag and drop wiring
          ['job', 'candidates'].forEach(kind => {
              const zone = document.getElementById(`${kind}Zone`);
              zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
              zone.addEventListener('dragleave', e => { e.preventDefault(); zone.classList.remove('dragover'); });
              zone.addEventListener('drop', e => {
                  e.preventDefault();
                  zone.classList.remove('dragover');
                  uploadFiles(e.dataTransfer.files, kind);
              });
              
              const input = document.getElementById(`${kind}Input`);
              input.addEventListener('change', () => uploadFiles(input.files, kind));
          });
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
async def upload_candidates(
    files: list[UploadFile] | None = File(None),
    file: UploadFile | None = File(None),
) -> dict[str, str | int]:
    """Upload one or more candidate datasets or resumes."""
    uploads = files or ([file] if file is not None else [])
    if not uploads:
        raise HTTPException(status_code=400, detail="Upload at least one candidate file.")

    saved_paths = []
    for upload in uploads:
        path = await _save_upload(upload, {"csv", "json", "pdf", "docx", "txt"})
        saved_paths.append(path)

    _state["candidates_paths"] = saved_paths
    return {"message": "candidates uploaded", "count": len(saved_paths)}


@app.post("/rank", response_model=RankingResponse)
def rank() -> RankingResponse:
    """Rank uploaded candidates against the uploaded job description."""
    job_path = _state.get("job_path")
    candidates_paths = _state.get("candidates_paths", [])
    
    if not job_path or not candidates_paths:
        raise HTTPException(status_code=400, detail="Upload a job file and candidate file(s) first.")
        
    try:
        pipeline = RankingPipeline(settings)
        ranking_output = pipeline.rank_from_files(job_path, candidates_paths)
        if len(ranking_output) == 3:
            results, output_path, job = ranking_output
        else:
            results, output_path = ranking_output
            job = None
        
        # Save state for chatbot
        _state["job"] = job
        _state["results"] = results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
        
    return RankingResponse(results=results, output_file=str(output_path))


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, str]:
    """Ask questions about the ranking results."""
    results = _state.get("results")
    if not results:
        raise HTTPException(status_code=400, detail="No ranking results available. Please rank candidates first.")
        
    job = _state.get("job")
    llm = GeminiRecruiterClient(settings)
    answer = llm.chat_about_results(req.question, results, job)
    return {"answer": answer}



@app.get("/results")
def results():
    """Download the latest ranked candidate CSV."""
    output_path = settings.output_dir / "ranked_candidates.csv"
    if not output_path.exists():
        return {"message": "No results available yet. Run /rank first."}
    return FileResponse(output_path, media_type="text/csv", filename="ranked_candidates.csv")


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
