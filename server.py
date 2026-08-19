# server.py
"""
Backend FastAPI do RAG Q&A System.

Substitui a interface Streamlit (app.py) por uma API REST consumida pelo
frontend estático em static/ (HTML/CSS/JS, visual inspirado no Claude.ai).

Rodar:
    python server.py
    # ou
    uvicorn server:app --reload --port 8000

Abra http://localhost:8000
"""
import os
import sys
import time
import logging
import sqlite3
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from src.database.models import RAGDatabase
from src.cache.cache_manager import CacheManager
from src.utils import list_documents

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("server")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DOCS_DIR = "data/documents"

app = FastAPI(title="RAG Q&A System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Modelos e modos disponíveis (mesmos do app.py Streamlit) ─────────────────
AVAILABLE_MODELS = {
    'mistral': {
        'label': 'Mistral 7B',
        'icon': '⚡',
        'desc': 'Rápido e eficiente. Ótimo para perguntas diretas.',
        'speed': 'Rápido',
    },
    'llama2': {
        'label': 'Llama 2 7B',
        'icon': '🦙',
        'desc': 'Meta AI. Boa cobertura geral e raciocínio.',
        'speed': 'Médio',
    },
    'neural-chat': {
        'label': 'Neural Chat',
        'icon': '🧠',
        'desc': 'Otimizado para conversas e explicações detalhadas.',
        'speed': 'Médio',
    },
    'openchat': {
        'label': 'OpenChat 3.5',
        'icon': '💬',
        'desc': 'Leve e rápido. Bom para respostas concisas.',
        'speed': 'Muito rápido',
    },
    'deepseek-r1': {
        'label': 'DeepSeek R1',
        'icon': '🔍',
        'desc': 'Excelente raciocínio e análise de documentos.',
        'speed': 'Lento',
    },
}

KNOWLEDGE_MODES = {
    'docs_only': {
        'label': 'Apenas documentos',
        'desc': 'Responde somente com base nos documentos carregados. Mais preciso e confiável.',
        'icon': '📄',
    },
    'hybrid': {
        'label': 'Documentos + Conhecimento geral',
        'desc': 'Usa documentos como prioridade e complementa com conhecimento geral da IA.',
        'icon': '🌐',
    },
}


# ─── Estado global do processo (equivalente ao st.session_state) ─────────────
class AppState:
    def __init__(self):
        self.rag_system = None
        self.db = RAGDatabase()
        self.cache = CacheManager()
        self.selected_model = 'mistral'
        self.knowledge_mode = 'docs_only'
        self.feedback_submitted = set()


state = AppState()


# ─── Schemas ──────────────────────────────────────────────────────────────────
class InitializeRequest(BaseModel):
    model_name: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    knowledge_mode: Optional[str] = None


class FeedbackRequest(BaseModel):
    interaction_id: int
    is_helpful: Optional[bool] = None
    comment: Optional[str] = None


class ScrapeRequest(BaseModel):
    url: str


class ModelChangeRequest(BaseModel):
    model: str


class ModeChangeRequest(BaseModel):
    mode: str


# ─── Status / configuração ────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    status = state.rag_system.get_status() if state.rag_system else None
    docs = list_documents(DOCS_DIR)
    return {
        "initialized": state.rag_system is not None,
        "system_ready": bool(status and status["system_ready"]),
        "vector_store_ready": bool(status and status["vector_store_ready"]),
        "model": state.selected_model,
        "knowledge_mode": state.knowledge_mode,
        "documents_count": len(docs),
        "documents": [os.path.basename(d) for d in docs],
    }


@app.get("/api/models")
def get_models():
    return {"models": AVAILABLE_MODELS, "selected": state.selected_model}

@app.get("/api/models/check")
def check_installed_models():
    """Verifica quais modelos estão instalados no Ollama."""
    import subprocess
    
    installed = []
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Pular header
            for line in lines[1:]:
                if line.strip():
                    model_name = line.split()[0].split(":")[0]  # Remove tag :latest
                    installed.append(model_name.lower())
    
    except Exception as e:
        logger.warning(f"Não foi possível verificar modelos Ollama: {e}")
    
    # Mapear quais modelos disponíveis estão instalados
    models_status = {}
    for key, data in AVAILABLE_MODELS.items():
        models_status[key] = {
            **data,
            "installed": any(key in m for m in installed) or any(m in key for m in installed)
        }
    
    return {
        "models": models_status,
        "installed_raw": installed,
        "selected": state.selected_model
    }


@app.get("/api/knowledge-modes")
def get_modes():
    return {"modes": KNOWLEDGE_MODES, "selected": state.knowledge_mode}


@app.post("/api/initialize")
def initialize(req: InitializeRequest):
    try:
        from src.rag_system import RAGSystem  # import pesado só quando necessário
        model = req.model_name or state.selected_model
        state.rag_system = RAGSystem(model_name=model)
        state.selected_model = model
        return {"success": True, "status": state.rag_system.get_status()}
    except Exception as e:
        logger.exception("Erro ao inicializar o sistema")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/build")
def build_knowledge_base():
    if not state.rag_system:
        raise HTTPException(status_code=400, detail="Inicialize o sistema primeiro.")
    success = state.rag_system.build_knowledge_base()
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Falha ao construir a base. Verifique se há documentos em data/documents/."
        )
    return {"success": True, "status": state.rag_system.get_status()}


# ─── Documentos ────────────────────────────────────────────────────────────────
@app.get("/api/documents")
def get_documents():
    docs = list_documents(DOCS_DIR)
    return {"documents": [os.path.basename(d) for d in docs], "count": len(docs)}


@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    allowed = {".pdf", ".txt", ".docx", ".md"}
    dest_dir = Path(DOCS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved, rejected = [], []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed:
            rejected.append(f.filename)
            continue
        content = await f.read()
        (dest_dir / f.filename).write_bytes(content)
        saved.append(f.filename)

    return {"saved": saved, "rejected": rejected, "count": len(saved)}


@app.post("/api/scrape")
def scrape_url(req: ScrapeRequest):
    if not state.rag_system:
        raise HTTPException(status_code=400, detail="Inicialize o sistema primeiro.")
    try:
        from src.scraper.web_scraper import WebScraper
        scraper = WebScraper()
        data = scraper.scrape_url(req.url)
        if not data:
            raise HTTPException(status_code=422, detail="Não foi possível extrair conteúdo dessa URL.")
        fp = scraper.save_to_document(data, documents_dir=DOCS_DIR)
        return {"success": True, "file": os.path.basename(fp)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Configurações ─────────────────────────────────────────────────────────────
@app.post("/api/settings/model")
def change_model(req: ModelChangeRequest):
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Modelo desconhecido.")
    state.selected_model = req.model
    if state.rag_system:
        try:
            state.rag_system.change_model(req.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Modelo salvo, mas falha ao trocar em runtime: {e}")
    return {"success": True, "model": req.model}


@app.post("/api/settings/mode")
def change_mode(req: ModeChangeRequest):
    if req.mode not in KNOWLEDGE_MODES:
        raise HTTPException(status_code=400, detail="Modo desconhecido.")
    state.knowledge_mode = req.mode
    if state.rag_system:
        try:
            state.rag_system.change_knowledge_mode(req.mode)
        except Exception:
            pass
    return {"success": True, "mode": req.mode}


# ─── Query (com cache semântico) ──────────────────────────────────────────────
@app.post("/api/query")
def query(req: QueryRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pergunta vazia.")
    if not state.rag_system or not state.rag_system.get_status()["system_ready"]:
        raise HTTPException(
            status_code=400,
            detail="Sistema não está pronto. Inicialize e construa a base de conhecimento primeiro."
        )

    mode = req.knowledge_mode or state.knowledge_mode

    # Cache semântico: reaproveita respostas de perguntas com sentido equivalente
    cached = state.cache.get_semantic(question)

    if cached:
        answer = cached["answer"]
        sources = cached["sources"]
        response_time = 0.0
        from_cache = True
        similarity = cached.get("similarity")
        model_label = "cache"
    else:
        start = time.time()
        result = state.rag_system.query(question, knowledge_mode=mode)
        response_time = time.time() - start
        answer = result["answer"]
        sources = result["sources"]
        state.cache.set(question=question, answer=answer, sources=sources)
        from_cache = False
        similarity = None
        model_label = state.selected_model

    interaction_id = state.db.save_interaction(
        question=question, answer=answer, sources=sources, model_used=model_label
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "interaction_id": interaction_id,
        "response_time": response_time,
        "from_cache": from_cache,
        "similarity": similarity,
    }


# ─── Feedback ──────────────────────────────────────────────────────────────────
@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    has_rating  = req.is_helpful is not None
    has_comment = bool(req.comment and req.comment.strip())

    if not has_rating and not has_comment:
        raise HTTPException(status_code=400, detail="Adicione uma avaliação ou comentário.")

    # Verificar se interaction_id existe
    recent = state.db.get_recent_interactions(limit=999999)
    valid_ids = {row[0] for row in recent}
    if req.interaction_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Interação {req.interaction_id} não encontrada.")

    state.db.save_feedback(
        interaction_id=req.interaction_id,
        is_helpful=req.is_helpful,
        comment=(req.comment or "").strip(),
    )
    state.feedback_submitted.add(req.interaction_id)
    return {"success": True}

@app.post("/api/feedback")
async def save_feedback(request: FeedbackRequest):
    """Salvar feedback do usuário."""
    try:
        # Validar se interaction_id existe no banco
        recent = state.db.get_recent_interactions(limit=10000)
        valid_ids = {row[0] for row in recent}  # IDs existentes

        if request.interaction_id not in valid_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Interação {request.interaction_id} não encontrada"
            )

        # Validar que tem rating ou comentário
        has_rating  = request.is_helpful is not None
        has_comment = bool(request.comment and request.comment.strip())

        if not has_rating and not has_comment:
            raise HTTPException(
                status_code=400,
                detail="Forneça uma avaliação ou comentário"
            )

        state.db.save_feedback(
            interaction_id=request.interaction_id,
            is_helpful=request.is_helpful,
            comment=request.comment
        )

        return {"status": "ok", "message": "Feedback salvo com sucesso"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao salvar feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Histórico ─────────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(limit: int = 20):
    rows = state.db.get_recent_interactions(limit=limit)
    return {
        "history": [
            {"id": r[0], "question": r[1], "answer": r[2], "timestamp": r[3]}
            for r in reversed(rows)
        ]
    }


# ─── Cache ─────────────────────────────────────────────────────────────────────
@app.get("/api/cache/stats")
def cache_stats():
    return state.cache.get_statistics()


@app.post("/api/cache/clear")
def cache_clear():
    state.cache.clear()
    return {"success": True}


# ─── Aprendizado contínuo ──────────────────────────────────────────────────────
@app.get("/api/learning/summary")
def learning_summary():
    try:
        from src.learning.learning_store import LearningStore
        store = LearningStore()
        summary = store.get_summary()
        pending = state.db.get_unprocessed_feedback_count()
        return {
            "available": True,
            "rules_count": summary["rules_count"],
            "examples_count": summary["examples_count"],
            "pending_feedback": pending,
            "rules": store.load_rules(),
            "examples": store.load_examples(limit=5),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/api/learning/analyze")
def learning_analyze():
    try:
        from src.learning.feedback_learner import FeedbackLearner
        learner = FeedbackLearner(ollama_model=state.selected_model)
        result = learner.analyze_feedback()
        if state.rag_system and state.rag_system.llm_chain:
            state.rag_system.llm_chain.reload_learning()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Dashboard / estatísticas ──────────────────────────────────────────────────
def _query_db(sql: str, params=()):
    conn = sqlite3.connect(state.db.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@app.get("/api/dashboard")
def dashboard():
    stats = state.db.get_statistics()
    cache_statistics = state.cache.get_statistics()

    feedback_over_time = _query_db("""
        SELECT DATE(f.timestamp) AS day,
               SUM(CASE WHEN f.is_helpful = 1 THEN 1 ELSE 0 END) AS helpful,
               SUM(CASE WHEN f.is_helpful = 0 THEN 1 ELSE 0 END) AS not_helpful
        FROM feedback f
        WHERE f.is_helpful IS NOT NULL
        GROUP BY day ORDER BY day
    """)
    questions_per_day = _query_db("""
        SELECT DATE(timestamp) AS day, COUNT(*) AS total
        FROM interactions GROUP BY day ORDER BY day
    """)
    models_used = _query_db("""
        SELECT model_used, COUNT(*) AS total FROM interactions
        GROUP BY model_used ORDER BY total DESC
    """)
    recent = _query_db("""
        SELECT SUBSTR(question,1,80) AS question, SUBSTR(answer,1,140) AS answer,
               model_used, timestamp
        FROM interactions ORDER BY timestamp DESC LIMIT 8
    """)

    return {
        "kpis": stats,
        "cache": cache_statistics,
        "feedback_over_time": feedback_over_time,
        "questions_per_day": questions_per_day,
        "models_used": models_used,
        "recent_interactions": recent,
    }


# ─── Frontend estático ─────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
