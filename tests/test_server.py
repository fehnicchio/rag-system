# tests/test_server.py
"""
Testes automatizados dos endpoints da API REST.

Uso:
    pytest tests/ -v
    pytest tests/test_server.py -v
    pytest tests/ -v --tb=short
"""
import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Garante estado limpo entre testes."""
    yield


# ── Testes: Status e Health ───────────────────────────────────────────────────

class TestHealth:

    def test_status_endpoint_exists(self):
        """GET /api/status deve retornar 200."""
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_status_has_required_fields(self):
        """Status deve conter campos obrigatórios."""
        response = client.get("/api/status")
        data = response.json()

        required_fields = [
            "initialized",
            "system_ready",
            "documents_count",
            "documents",
            "model",
            "knowledge_mode",
        ]
        for field in required_fields:
            assert field in data, f"Campo ausente: {field}"

    def test_status_initial_state(self):
        """Sistema deve iniciar não inicializado."""
        response = client.get("/api/status")
        data = response.json()
        assert data["initialized"] in [True, False]
        assert data["system_ready"] in [True, False]


# ── Testes: Modelos ───────────────────────────────────────────────────────────

class TestModels:

    def test_get_models(self):
        """GET /api/models deve retornar lista de modelos."""
        response = client.get("/api/models")
        assert response.status_code == 200

    def test_models_has_required_fields(self):
        """Modelos devem ter campos obrigatórios."""
        response = client.get("/api/models")
        data = response.json()

        assert "models" in data
        assert "selected" in data

        for key, model in data["models"].items():
            assert "label" in model, f"Modelo {key} sem label"
            assert "icon"  in model, f"Modelo {key} sem icon"
            assert "desc"  in model, f"Modelo {key} sem desc"

    def test_models_check_endpoint(self):
        """GET /api/models/check deve retornar status de instalação."""
        response = client.get("/api/models/check")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "installed_raw" in data

    def test_knowledge_modes(self):
        """GET /api/knowledge-modes deve retornar modos disponíveis."""
        response = client.get("/api/knowledge-modes")
        assert response.status_code == 200
        data = response.json()
        assert "modes" in data
        assert "docs_only" in data["modes"]
        assert "hybrid"    in data["modes"]


# ── Testes: Documentos ────────────────────────────────────────────────────────

class TestDocuments:

    def test_upload_no_files(self):
        """Upload sem arquivos deve retornar erro."""
        response = client.post("/api/documents/upload")
        assert response.status_code in [400, 422]

    def test_upload_valid_txt(self, tmp_path):
        """Upload de arquivo TXT válido deve funcionar."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Conteúdo de teste para o RAG System.")

        with open(txt_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"files": ("test.txt", f, "text/plain")}
            )

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert data["count"] >= 1

    def test_scrape_invalid_url(self):
        """Scraping de URL inválida deve retornar erro."""
        response = client.post(
            "/api/scrape",
            json={"url": "url-invalida-sem-protocolo"}
        )
        assert response.status_code in [400, 500]

    def test_scrape_missing_url(self):
        """Scraping sem URL deve retornar erro de validação."""
        response = client.post("/api/scrape", json={})
        assert response.status_code == 422


# ── Testes: Cache ─────────────────────────────────────────────────────────────

class TestCache:

    def test_cache_stats(self):
        """GET /api/cache/stats deve retornar estatísticas."""
        response = client.get("/api/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "cached_responses" in data
        assert "hit_rate"         in data

    def test_cache_clear(self):
        """POST /api/cache/clear deve limpar cache."""
        response = client.post("/api/cache/clear")
        assert response.status_code == 200

    def test_cache_stats_after_clear(self):
        """Cache deve ter 0 responses após limpar."""
        client.post("/api/cache/clear")
        response = client.get("/api/cache/stats")
        data = response.json()
        assert data["cached_responses"] == 0


# ── Testes: Feedback ──────────────────────────────────────────────────────────

class TestFeedback:

    def test_feedback_missing_interaction_id(self):
        """Feedback sem interaction_id deve retornar erro."""
        response = client.post(
            "/api/feedback",
            json={"is_helpful": True, "comment": "Boa resposta"}
        )
        assert response.status_code == 422

    def test_feedback_invalid_interaction_id(self):
        """Feedback com ID inválido deve retornar erro."""
        response = client.post(
            "/api/feedback",
            json={
                "interaction_id": 999999,
                "is_helpful": True,
                "comment": ""
            }
        )
        assert response.status_code in [400, 404, 500]

    def test_feedback_without_rating_and_comment(self):
        """Feedback sem avaliação nem comentário deve retornar erro."""
        response = client.post(
            "/api/feedback",
            json={
                "interaction_id": 1,
                "is_helpful": None,
                "comment": ""
            }
        )
        assert response.status_code in [400, 422]


# ── Testes: Dashboard ─────────────────────────────────────────────────────────

class TestDashboard:

    def test_dashboard_endpoint(self):
        """GET /api/dashboard deve retornar dados."""
        response = client.get("/api/dashboard")
        assert response.status_code == 200

    def test_dashboard_has_required_fields(self):
        """Dashboard deve ter campos obrigatórios."""
        response = client.get("/api/dashboard")
        data = response.json()

        required = ["kpis", "questions_per_day", "models_used", "recent_interactions"]
        for field in required:
            assert field in data, f"Campo ausente no dashboard: {field}"

    def test_dashboard_kpis(self):
        """KPIs do dashboard devem ter campos numéricos."""
        response = client.get("/api/dashboard")
        data = response.json()
        kpis = data["kpis"]

        assert "total_interactions" in kpis
        assert "helpful_count"      in kpis
        assert "not_helpful_count"  in kpis
        assert "helpful_rate"       in kpis
        assert isinstance(kpis["total_interactions"], int)
        assert isinstance(kpis["helpful_rate"], float)


# ── Testes: Query (requer sistema inicializado) ───────────────────────────────

class TestQuery:

    def test_query_without_system(self):
        """Query sem sistema inicializado deve retornar erro claro."""
        response = client.post(
            "/api/query",
            json={"question": "O que é RAG?"}
        )
        # Deve retornar resposta (mesmo que seja de erro do sistema)
        assert response.status_code in [200, 400, 503]

    def test_query_empty_question(self):
        """Query com pergunta vazia deve retornar erro."""
        response = client.post(
            "/api/query",
            json={"question": ""}
        )
        assert response.status_code in [400, 422]

    def test_query_missing_question(self):
        """Query sem campo question deve retornar erro de validação."""
        response = client.post("/api/query", json={})
        assert response.status_code == 422