# src/learning/learning_store.py
"""
Persistência do conhecimento aprendido via feedback.

Salva dois tipos de dados em data/learning/:
  - rules.json    → regras de comportamento extraídas dos comentários negativos
  - examples.json → pares (pergunta, resposta corrigida) para few-shot prompting
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LEARNING_DIR = Path("data/learning")


class LearningStore:
    """Lê e grava o conhecimento aprendido em disco."""

    def __init__(self, learning_dir: str | Path = DEFAULT_LEARNING_DIR):
        self.dir = Path(learning_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.rules_path    = self.dir / "rules.json"
        self.examples_path = self.dir / "examples.json"

    # ── Regras ───────────────────────────────────────────────────────────────

    def load_rules(self) -> list[dict]:
        """
        Retorna lista de regras ativas.

        Formato de cada regra:
          {
            "rule":        "Seja mais direto e objetivo nas respostas",
            "source":      "Comentário do usuário: 'resposta muito longa'",
            "created_at":  "2024-01-15T10:30:00",
            "active":      true
          }
        """
        if not self.rules_path.exists():
            return []
        try:
            data = json.loads(self.rules_path.read_text(encoding="utf-8"))
            return [r for r in data if r.get("active", True)]
        except Exception as e:
            logger.error(f"Erro ao carregar regras: {e}")
            return []

    def save_rules(self, rules: list[dict]) -> None:
        """Sobrescreve o arquivo de regras."""
        try:
            self.rules_path.write_text(
                json.dumps(rules, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(f"✅ {len(rules)} regra(s) salva(s) em {self.rules_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar regras: {e}")

    def add_rules(self, new_rules: list[str], source: str = "") -> None:
        """
        Adiciona novas regras sem duplicar as já existentes.

        Args:
            new_rules: lista de strings com as regras
            source:    texto de onde veio a regra (para rastreabilidade)
        """
        existing = self.load_rules()
        existing_texts = {r["rule"].strip().lower() for r in existing}

        added = 0
        for rule_text in new_rules:
            if rule_text.strip().lower() not in existing_texts:
                existing.append({
                    "rule":       rule_text.strip(),
                    "source":     source,
                    "created_at": datetime.now().isoformat(),
                    "active":     True,
                })
                existing_texts.add(rule_text.strip().lower())
                added += 1

        self.save_rules(existing)
        logger.info(f"  {added} nova(s) regra(s) adicionada(s) ({len(existing)} total)")

    def deactivate_rule(self, rule_text: str) -> None:
        """Desativa uma regra pelo texto (sem excluir do arquivo)."""
        rules = self.load_rules()
        for r in rules:
            if r["rule"].strip().lower() == rule_text.strip().lower():
                r["active"] = False
        self.save_rules(rules)

    # ── Exemplos few-shot ─────────────────────────────────────────────────────

    def load_examples(self, limit: int = 5) -> list[dict]:
        """
        Retorna os exemplos de Q&A corrigidos mais recentes.

        Formato de cada exemplo:
          {
            "question":         "O que é visão computacional?",
            "bad_answer":       "...",          # resposta original ruim
            "good_answer":      "...",          # versão corrigida pelo LLM
            "feedback_comment": "muito vaga",   # comentário original do usuário
            "created_at":       "2024-01-15T..."
          }
        """
        if not self.examples_path.exists():
            return []
        try:
            data = json.loads(self.examples_path.read_text(encoding="utf-8"))
            # Mais recentes primeiro
            data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return data[:limit]
        except Exception as e:
            logger.error(f"Erro ao carregar exemplos: {e}")
            return []

    def save_examples(self, examples: list[dict]) -> None:
        """Sobrescreve o arquivo de exemplos."""
        try:
            self.examples_path.write_text(
                json.dumps(examples, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(f"✅ {len(examples)} exemplo(s) salvo(s) em {self.examples_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar exemplos: {e}")

    def add_examples(self, new_examples: list[dict]) -> None:
        """
        Adiciona novos exemplos, evitando duplicatas pela pergunta.
        Mantém no máximo 50 exemplos (descarta os mais antigos).
        """
        MAX_EXAMPLES = 50
        existing = self.load_examples(limit=MAX_EXAMPLES)
        existing_questions = {e["question"].strip().lower() for e in existing}

        added = 0
        for ex in new_examples:
            q = ex.get("question", "").strip().lower()
            if q and q not in existing_questions:
                ex["created_at"] = datetime.now().isoformat()
                existing.append(ex)
                existing_questions.add(q)
                added += 1

        # Manter apenas os MAX_EXAMPLES mais recentes
        existing.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        self.save_examples(existing[:MAX_EXAMPLES])
        logger.info(f"  {added} novo(s) exemplo(s) adicionado(s) ({len(existing)} total)")

    # ── Utilitários ───────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        return {
            "rules_count":    len(self.load_rules()),
            "examples_count": len(self.load_examples(limit=100)),
            "rules_path":     str(self.rules_path),
            "examples_path":  str(self.examples_path),
        }

    def clear_all(self) -> None:
        """Remove todo o aprendizado (use com cautela)."""
        for path in [self.rules_path, self.examples_path]:
            if path.exists():
                path.unlink()
        logger.warning("⚠️  Todo o aprendizado foi removido.")