# AI-Powered Candidate Ranking System

## 1. Problem
- Recruiters receive too many resumes to review manually.
- Keyword search misses semantically relevant candidates.
- Screening decisions need explainability and consistency.

## 2. Solution
- Parse job descriptions and candidate profiles.
- Use semantic embeddings and vector search.
- Combine semantic relevance with skills, experience, education, behaviour, and activity signals.
- Use Gemini LLM reranking for recruiter-style final notes.

## 3. Architecture
- FastAPI service.
- File parsers for TXT, PDF, and DOCX.
- Candidate ingestion from CSV and JSON.
- Sentence Transformers embeddings.
- FAISS vector index with NumPy fallback.
- Hybrid ranking engine.
- CSV export and API results.

## 4. Pipeline
- Upload JD.
- Extract structured requirements.
- Upload candidate dataset.
- Normalize candidate evidence.
- Generate embeddings.
- Score and rank candidates.
- Export shortlist.

## 5. Embedding Flow
- Job requirements are converted into a semantic search query.
- Candidate summaries are embedded.
- FAISS retrieves candidates with highest cosine similarity.

## 6. LLM Flow
- Gemini extracts job requirements into JSON.
- Top 20 candidates are sent for expert recruiter reranking.
- The final response includes reasons, strengths, and weaknesses.

## 7. Ranking Formula
- 40% Semantic similarity.
- 20% Skill match.
- 15% Experience match.
- 10% Education match.
- 10% Behaviour match.
- 5% Platform activity.

## 8. Demo
- Start API with `uvicorn app.api.main:app --reload`.
- Open `http://127.0.0.1:8000`.
- Upload sample JD and candidates.
- Run ranking and download CSV.

## 9. Results
- Ranked shortlist.
- Score breakdown.
- Recruiter notes.
- Confidence score.

## 10. Future Scope
- Auth and role-based recruiter dashboards.
- Persistent database.
- Bias and fairness monitoring.
- Human feedback loop.
- ATS integrations.
