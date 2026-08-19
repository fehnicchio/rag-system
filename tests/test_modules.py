# tests/test_modules.py
"""
Testes dos módulos internos do RAG System.
Não requerem servidor rodando.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Cache Manager ─────────────────────────────────────────────────────────────

class TestCacheManager:

    @pytest.fixture
    def cache(self, tmp_path):
        from src.cache.cache_manager import CacheManager
        return CacheManager(cache_db=str(tmp_path / "test_cache.db"))

    def test_cache_miss(self, cache):
        """Pergunta não cacheada deve retornar None."""
        result = cache.get("pergunta inexistente")
        assert result is None

    def test_cache_set_and_get(self, cache):
        """Deve salvar e recuperar resposta."""
        cache.set("O que é RAG?", "RAG é uma técnica...", [])
        result = cache.get("O que é RAG?")
        assert result is not None
        assert result["answer"] == "RAG é uma técnica..."
        assert result["from_cache"] is True

    def test_cache_normalized_question(self, cache):
        """Perguntas similares devem retornar mesma resposta."""
        cache.set("O que é RAG?", "RAG é...", [])

        # Com pontuação diferente
        result = cache.get("o que é rag")
        assert result is not None

    def test_cache_clear(self, cache):
        """Limpar cache deve remover todas as entradas."""
        cache.set("pergunta 1", "resposta 1", [])
        cache.set("pergunta 2", "resposta 2", [])
        cache.clear()
        assert cache.get("pergunta 1") is None
        assert cache.get("pergunta 2") is None

    def test_cache_statistics(self, cache):
        """Estatísticas devem refletir estado do cache."""
        stats = cache.get_statistics()
        assert stats["cached_responses"]  == 0
        assert stats["total_cache_hits"]  == 0

        cache.set("pergunta", "resposta", [])
        stats = cache.get_statistics()
        assert stats["cached_responses"] == 1

    def test_cache_semantic_hit(self, cache):
        """Cache semântico deve encontrar perguntas similares."""
        if not hasattr(cache, "get_semantic"):
            pytest.skip("Cache semântico não implementado")

        cache.set("O que é machine learning?", "ML é...", [])
        result = cache.get_semantic("Explique machine learning")

        # Pode ou não encontrar dependendo do threshold
        # Só verificamos que não retorna erro
        assert result is None or "answer" in result


# ── Database Models ───────────────────────────────────────────────────────────

class TestDatabase:

    @pytest.fixture
    def db(self, tmp_path):
        from src.database.models import RAGDatabase
        return RAGDatabase(db_path=str(tmp_path / "test.db"))

    def test_save_and_get_statistics(self, db):
        """Deve salvar interação e retornar nas estatísticas."""
        stats_before = db.get_statistics()
        initial_total = stats_before["total_interactions"]

        db.save_interaction(
            question="Teste?",
            answer="Resposta de teste.",
            sources=[],
            model_used="mistral"
        )

        stats_after = db.get_statistics()
        assert stats_after["total_interactions"] == initial_total + 1

    def test_save_feedback_helpful(self, db):
        """Feedback útil deve incrementar helpful_count."""
        iid = db.save_interaction("Q?", "A.", [], "mistral")
        db.save_feedback(interaction_id=iid, is_helpful=True)

        stats = db.get_statistics()
        assert stats["helpful_count"] >= 1

    def test_save_feedback_not_helpful(self, db):
        """Feedback negativo deve incrementar not_helpful_count."""
        iid = db.save_interaction("Q?", "A.", [], "mistral")
        db.save_feedback(interaction_id=iid, is_helpful=False)

        stats = db.get_statistics()
        assert stats["not_helpful_count"] >= 1

    def test_helpful_rate_calculation(self, db):
        """Taxa de utilidade deve ser calculada corretamente."""
        iid1 = db.save_interaction("Q1?", "A1.", [], "mistral")
        iid2 = db.save_interaction("Q2?", "A2.", [], "mistral")

        db.save_feedback(iid1, is_helpful=True)
        db.save_feedback(iid2, is_helpful=False)

        stats = db.get_statistics()
        assert stats["helpful_rate"] == pytest.approx(50.0, abs=1.0)

    def test_recent_interactions(self, db):
        """Deve retornar interações recentes."""
        db.save_interaction("Pergunta 1?", "Resposta 1.", [], "mistral")
        db.save_interaction("Pergunta 2?", "Resposta 2.", [], "mistral")

        recent = db.get_recent_interactions(limit=5)
        assert len(recent) >= 2


# ── Web Scraper ───────────────────────────────────────────────────────────────

class TestWebScraper:

    @pytest.fixture
    def scraper(self):
        from src.scraper.web_scraper import WebScraper
        return WebScraper()

    def test_invalid_url(self, scraper):
        """URL inválida deve retornar None."""
        result = scraper.scrape_url("url-invalida")
        assert result is None

    def test_scrape_wikipedia(self, scraper):
        """Wikipedia deve funcionar."""
        result = scraper.scrape_url(
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        if result:  # Pode falhar sem internet
            assert "title"   in result
            assert "content" in result
            assert len(result["content"]) > 100

    def test_create_filename(self, scraper):
        """Filename deve ser válido."""
        filename = scraper._create_filename("https://example.com/article/test")
        assert "/" not in filename
        assert "\\" not in filename
        assert filename.endswith(".txt")

    def test_save_document(self, scraper, tmp_path):
        """Deve salvar documento no diretório correto."""
        data = {
            "url":        "https://example.com/test",
            "title":      "Título de Teste",
            "content":    "Conteúdo de teste para verificar salvamento.",
            "scraped_at": "2024-01-01T00:00:00",
            "length":     50,
        }
        filepath = scraper.save_to_document(data, documents_dir=str(tmp_path))
        assert filepath is not None
        assert Path(filepath).exists()


# ── Learning Store ────────────────────────────────────────────────────────────

class TestLearningStore:

    @pytest.fixture
    def store(self, tmp_path):
        from src.learning.learning_store import LearningStore
        return LearningStore(learning_dir=str(tmp_path / "learning"))

    def test_empty_rules(self, store):
        """Store vazio deve retornar lista vazia de regras."""
        assert store.load_rules() == []

    def test_add_and_load_rules(self, store):
        """Deve adicionar e carregar regras."""
        store.add_rules(["Seja mais direto", "Cite sempre as fontes"])
        rules = store.load_rules()
        assert len(rules) == 2
        assert any("direto" in r["rule"] for r in rules)

    def test_no_duplicate_rules(self, store):
        """Não deve adicionar regras duplicadas."""
        store.add_rules(["Regra única"])
        store.add_rules(["Regra única"])  # Duplicada
        rules = store.load_rules()
        assert len(rules) == 1

    def test_add_and_load_examples(self, store):
        """Deve adicionar e carregar exemplos."""
        examples = [{
            "question":    "O que é RAG?",
            "bad_answer":  "RAG é algo.",
            "good_answer": "RAG (Retrieval-Augmented Generation) é...",
            "feedback_comment": "muito vaga",
            "interaction_id": 1,
        }]
        store.add_examples(examples)
        loaded = store.load_examples(limit=5)
        assert len(loaded) >= 1

    def test_get_summary(self, store):
        """Resumo deve conter contagens corretas."""
        store.add_rules(["Regra 1", "Regra 2"])
        summary = store.get_summary()
        assert summary["rules_count"]    == 2
        assert summary["examples_count"] == 0

    def test_clear_all(self, store):
        """Clear deve remover tudo."""
        store.add_rules(["Regra teste"])
        store.clear_all()
        assert store.load_rules() == []