# src/learning/feedback_learner.py
"""
FeedbackLearner — aprende com feedbacks negativos para melhorar respostas futuras.

Estratégia dual:
  1. REGRAS DE COMPORTAMENTO
     O LLM lê os comentários negativos e extrai instruções gerais
     ("Seja mais direto", "Sempre cite a fonte", etc.).
     Essas regras são injetadas no system prompt de toda resposta futura.

  2. EXEMPLOS FEW-SHOT (Q&A corrigidos)
     Para cada feedback negativo com comentário, o LLM gera uma versão
     melhorada da resposta original. Esses pares (pergunta → resposta boa)
     são injetados como exemplos no prompt para guiar o modelo.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackLearner:
    """Analisa feedbacks negativos e atualiza o aprendizado do sistema."""

    # Quantos feedbacks negativos processar por rodada de análise
    BATCH_SIZE = 20

    def __init__(
        self,
        db_path: str = "data/rag_feedback.db",
        learning_dir: str = "data/learning",
        ollama_model: str = "mistral",
    ):
        self.db_path = db_path
        self.ollama_model = ollama_model

        from src.learning.learning_store import LearningStore
        self.store = LearningStore(learning_dir)

        self._llm = None  # lazy init

    # ── LLM (lazy) ───────────────────────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            try:
                from langchain_ollama import OllamaLLM
                self._llm = OllamaLLM(model=self.ollama_model, temperature=0.2)
                logger.info(f"✅ LLM carregado para aprendizado ({self.ollama_model})")
            except Exception as e:
                raise RuntimeError(f"Não foi possível carregar o LLM Ollama: {e}")
        return self._llm

    # ── Leitura do banco ─────────────────────────────────────────────────────

    def _get_negative_feedback(self) -> list[dict]:
        """
        Retorna feedbacks negativos com comentário ainda não processados.
        Considera 'não processado' = sem entrada em feedback_processed.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Garantir tabela de controle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_processed (
                feedback_id INTEGER PRIMARY KEY,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        cursor.execute("""
            SELECT
                f.id          AS feedback_id,
                f.comment,
                f.timestamp   AS feedback_ts,
                i.id          AS interaction_id,
                i.question,
                i.answer,
                i.sources
            FROM feedback f
            JOIN interactions i ON i.id = f.interaction_id
            LEFT JOIN feedback_processed fp ON fp.feedback_id = f.id
            WHERE f.is_helpful = 0          -- apenas negativos
              AND f.comment IS NOT NULL
              AND TRIM(f.comment) != ''     -- com comentário real
              AND fp.feedback_id IS NULL    -- ainda não processados
            ORDER BY f.timestamp DESC
            LIMIT ?
        """, (self.BATCH_SIZE,))

        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        logger.info(f"📋 {len(rows)} feedback(s) negativo(s) não processado(s) encontrado(s)")
        return rows

    def _mark_as_processed(self, feedback_ids: list[int]) -> None:
        if not feedback_ids:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO feedback_processed (feedback_id) VALUES (?)",
            [(fid,) for fid in feedback_ids]
        )
        conn.commit()
        conn.close()

    # ── Extração de regras ───────────────────────────────────────────────────

    def _extract_rules(self, feedbacks: list[dict]) -> list[str]:
        """
        Pede ao LLM para extrair regras gerais de comportamento
        a partir dos comentários negativos.
        """
        if not feedbacks:
            return []

        comments_block = "\n".join(
            f"- \"{f['comment']}\"" for f in feedbacks
        )

        prompt = f"""Você é um especialista em qualidade de sistemas de perguntas e respostas.

Abaixo estão comentários de usuários insatisfeitos com as respostas de um assistente de IA:

{comments_block}

Com base nesses comentários, extraia REGRAS GERAIS DE COMPORTAMENTO que o assistente deve seguir
para melhorar suas respostas futuras.

INSTRUÇÕES:
- Escreva cada regra como uma instrução direta e acionável (ex: "Sempre cite a fonte do documento")
- Seja específico, não genérico (evite "Melhore as respostas")
- Máximo de 8 regras
- Uma regra por linha, sem numeração, sem bullets
- Responda APENAS com as regras, sem explicações adicionais
- Escreva em português

REGRAS:"""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            rules = [
                line.strip().lstrip("-•*").strip()
                for line in text.strip().splitlines()
                if line.strip() and len(line.strip()) > 10
            ]
            logger.info(f"  📌 {len(rules)} regra(s) extraída(s) dos comentários")
            return rules

        except Exception as e:
            logger.error(f"Erro ao extrair regras: {e}")
            return []

    # ── Geração de exemplos corrigidos ───────────────────────────────────────

    def _generate_corrected_examples(self, feedbacks: list[dict]) -> list[dict]:
        """
        Para cada feedback negativo, pede ao LLM para gerar uma versão
        melhorada da resposta original, levando em conta o comentário do usuário.
        """
        examples = []

        for f in feedbacks:
            prompt = f"""Você é um especialista em melhoria de respostas de sistemas RAG.

PERGUNTA DO USUÁRIO:
{f['question']}

RESPOSTA ORIGINAL (que o usuário avaliou negativamente):
{f['answer']}

CRÍTICA DO USUÁRIO:
"{f['comment']}"

Reescreva a resposta corrigindo exatamente o problema apontado pelo usuário.
Mantenha as informações corretas, mas ajuste o estilo, profundidade ou formato conforme a crítica.
Responda APENAS com a resposta corrigida, sem explicações ou prefácios."""

            try:
                llm = self._get_llm()
                response = llm.invoke(prompt)
                good_answer = response.content if hasattr(response, "content") else str(response)

                if good_answer.strip():
                    examples.append({
                        "question":         f["question"],
                        "bad_answer":       f["answer"],
                        "good_answer":      good_answer.strip(),
                        "feedback_comment": f["comment"],
                        "interaction_id":   f["interaction_id"],
                    })
                    logger.info(f"  ✍️  Exemplo corrigido gerado para: '{f['question'][:60]}...'")

            except Exception as e:
                logger.error(f"Erro ao gerar exemplo para feedback {f['feedback_id']}: {e}")

        return examples

    # ── API pública ───────────────────────────────────────────────────────────

    def analyze_feedback(self) -> dict:
        """
        Pipeline completo de aprendizado:
          1. Busca feedbacks negativos não processados
          2. Extrai regras de comportamento
          3. Gera exemplos corrigidos (few-shot)
          4. Persiste tudo no LearningStore
          5. Marca feedbacks como processados

        Returns:
            dict com resumo do que foi aprendido
        """
        logger.info("=" * 50)
        logger.info("🧠 Iniciando análise de feedback")
        logger.info("=" * 50)

        feedbacks = self._get_negative_feedback()

        if not feedbacks:
            logger.info("ℹ️  Nenhum feedback novo para processar.")
            return {
                "processed": 0,
                "new_rules": [],
                "new_examples": 0,
                "store_summary": self.store.get_summary(),
            }

        # 1. Extrair regras
        logger.info("📌 Extraindo regras de comportamento...")
        new_rules = self._extract_rules(feedbacks)
        if new_rules:
            self.store.add_rules(
                new_rules,
                source=f"Análise de {len(feedbacks)} feedbacks em {datetime.now().strftime('%Y-%m-%d')}"
            )

        # 2. Gerar exemplos corrigidos
        logger.info("✍️  Gerando exemplos corrigidos...")
        new_examples = self._generate_corrected_examples(feedbacks)
        if new_examples:
            self.store.add_examples(new_examples)

        # 3. Marcar como processados
        self._mark_as_processed([f["feedback_id"] for f in feedbacks])

        summary = self.store.get_summary()
        logger.info("=" * 50)
        logger.info(f"✅ Aprendizado concluído!")
        logger.info(f"   Feedbacks processados : {len(feedbacks)}")
        logger.info(f"   Novas regras          : {len(new_rules)}")
        logger.info(f"   Novos exemplos        : {len(new_examples)}")
        logger.info(f"   Total regras ativas   : {summary['rules_count']}")
        logger.info(f"   Total exemplos        : {summary['examples_count']}")
        logger.info("=" * 50)

        return {
            "processed":    len(feedbacks),
            "new_rules":    new_rules,
            "new_examples": len(new_examples),
            "store_summary": summary,
        }

    def get_learning_summary(self) -> dict:
        """Retorna o estado atual do aprendizado sem processar novos feedbacks."""
        return self.store.get_summary()

    def reset_learning(self) -> None:
        """Remove todo o aprendizado acumulado (use com cautela)."""
        self.store.clear_all()
        logger.warning("⚠️  Aprendizado resetado.")


# ── Teste standalone ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    learner = FeedbackLearner()
    result = learner.analyze_feedback()

    print("\n📊 Resultado:")
    print(f"  Feedbacks processados: {result['processed']}")
    print(f"  Novas regras:          {len(result['new_rules'])}")
    print(f"  Novos exemplos:        {result['new_examples']}")
    print(f"\n  Estado do aprendizado: {result['store_summary']}")