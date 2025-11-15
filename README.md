# HackRx Policy Query Copilot

> Modern chatbot experience for interrogating insurance policy PDFs using FastAPI, FAISS, Firebase, and Google Gemini.

## ✨ Highlights

- **AI policy answers** powered by PyMuPDF parsing, FAISS retrieval, and Gemini completions.
- **Knowledge workspace** to ingest URLs, select docs, and read back metadata stored in Firebase.
- **Polished chat frontend** (HTML/CSS/JS) with voice/file inputs, light/dark themes, and live health badges.
- **Secure APIs** using bearer tokens, structured errors, and `/health` monitoring.

## 🧱 Tech stack

| Layer    | Tools |
|----------|-------|
| Backend  | FastAPI, Uvicorn, PyMuPDF, FAISS, Google Generative AI SDK, Firebase Admin |
| Frontend | HTML, CSS (Inter, responsive layout, theme toggle), Vanilla JS (fetch, voice capture, file upload) |
| Tooling  | Python 3.11, dotenv, Docker/Docker Compose (dev convenience) |

## 🗂️ Structure

```
policy_query/
├── hackrx_faiss_api.py      # FastAPI service + endpoints
├── hackrx-frontend/
│   ├── index.html           # Chat UI
│   ├── style.css            # Light/dark visual system
│   └── script.js            # Message flow, status chip, theme toggle
├── requirements.txt
├── Dockerfile               # Backend container image
├── docker-compose.yml       # api + frontend services
└── env_template.txt         # Sample .env
```

## 🚀 Getting started

### 1. Install dependencies
```bash
git clone https://github.com/<you>/policy_query.git
cd policy_query
python -m venv venv && venv\Scripts\activate  # or source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure env vars
Create `.env` in the repo root:
```env
HACKRX_API_KEY=your-policy-api-key
GEMINI_API_KEY=your-gemini-api-key
FIREBASE_PROJECT_ID=hackrx-dfb45
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/GOOGLE_APPLICATION_CREDENTIALS.json
```

### 3. Run locally
```bash
uvicorn hackrx_faiss_api:app --host 127.0.0.1 --port 8000
# Frontend: open hackrx-frontend/index.html using Live Server or any static server
```

### 4. Health check
```bash
curl http://127.0.0.1:8000/health
```

### Optional: Compose both services
```bash
docker compose up --build
# API available at http://localhost:8000, frontend at http://localhost:3000
```

## 📡 Core APIs

- `POST /ingest-url` – Fetches remote PDF/TXT, chunks & indexes it.
- `GET /kb/docs` – Lists stored documents and chunk counts for UI selection.
- `POST /ask-policy` – Takes a natural language question + doc filters, returns Gemini answer.
- `GET /health` – Simple status endpoint for the frontend badge.

## 🧪 Tips

- Use `/docs` (Swagger) for experimentation.
- The frontend “Save Token” stores bearer keys in session memory only.
- Voice and file buttons use browser APIs; ensure HTTPS in production.

## 📜 License

MIT © 2025. Contributions welcome!
