# src/generation/llm_chain.py
from langchain_ollama import OllamaLLM
import logging

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Cadeia de RAG (Retrieval + Generation) com aprendizado contínuo.

    A cada query, carrega do LearningStore:
      • Regras de comportamento  → adicionadas ao system prompt
      • Exemplos few-shot        → mostram ao modelo como responder bem
    Isso faz as correções dos usuários serem aplicadas automaticamente
    em todas as respostas futuras.
    """

    # Quantos exemplos few-shot injetar por prompt (evitar context overflow)
    MAX_FEW_SHOT_EXAMPLES = 3

    def __init__(self, retriever, api_key, temperature=0.7, max_tokens=500):
        """
        Args:
            retriever:    Retriever do FAISS (com reranking)
            api_key:      OpenAI API key (mantido por compatibilidade)
            temperature:  Criatividade do LLM (0–1)
            max_tokens:   Tamanho máximo da resposta
        """
        logger.info("Inicializando LLM...")

        try:
            self.llm = OllamaLLM(model="mistral", temperature=temperature)
            logger.info("✅ LLM Ollama carregado (Mistral)")
        except Exception as e:
            logger.error(f"Erro ao carregar LLM: {e}")
            raise

        self.retriever = retriever

        # Carregar LearningStore (sem falhar se ainda não existir)
        try:
            from src.learning.learning_store import LearningStore
            self.learning_store = LearningStore()
            logger.info("✅ LearningStore conectado ao RAGChain")
        except Exception as e:
            logger.warning(f"⚠️  LearningStore não disponível: {e}")
            self.learning_store = None

        logger.info("✅ RAG Chain pronto")

    # ── Construção do prompt ──────────────────────────────────────────────────

    def _build_rules_block(self) -> str:
        """
        Retorna o bloco de regras aprendidas para injetar no prompt.
        Vazio se não houver regras.
        """
        if not self.learning_store:
            return ""

        rules = self.learning_store.load_rules()
        if not rules:
            return ""

        rules_text = "\n".join(f"- {r['rule']}" for r in rules)
        return f"""
REGRAS DE COMPORTAMENTO APRENDIDAS (siga obrigatoriamente):
{rules_text}
"""

    def _build_examples_block(self) -> str:
        """
        Retorna o bloco de exemplos few-shot para injetar no prompt.
        Vazio se não houver exemplos.
        """
        if not self.learning_store:
            return ""

        examples = self.learning_store.load_examples(limit=self.MAX_FEW_SHOT_EXAMPLES)
        if not examples:
            return ""

        block = "\nEXEMPLOS DE RESPOSTAS DE QUALIDADE (baseados em correções de usuários reais):\n"
        for i, ex in enumerate(examples, 1):
            block += f"""
Exemplo {i}:
  Pergunta: {ex['question']}
  Resposta correta: {ex['good_answer']}
"""
        block += "\n"
        return block

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Monta o prompt completo com:
          1. Instrução base
          2. Regras aprendidas (se houver)
          3. Exemplos few-shot (se houver)
          4. Contexto dos documentos
          5. Pergunta
        """
        rules_block    = self._build_rules_block()
        examples_block = self._build_examples_block()

        has_learning = bool(rules_block or examples_block)
        if has_learning:
            logger.debug("🧠 Injetando aprendizado no prompt")

        return f"""Você é um assistente inteligente especializado em análise de documentos.
Priorize responder com base nos documentos fornecidos.
Apenas se a resposta NÃO estiver nos documentos, use seu conhecimento geral.
SEMPRE indique a fonte da informação.

Se a resposta está no documento:
"Segundo os documentos: [resposta] (Fonte: nome_do_doc)"

Se usou conhecimento geral:
"Nos documentos não encontrei, mas geralmente: [resposta]"
{rules_block}{examples_block}
CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA:
{question}

RESPOSTA:"""

    # ── Query principal ───────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Faz uma pergunta ao RAG, com aprendizado aplicado automaticamente.

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        try:
            logger.info(f"Processando pergunta: {question}")

            # 1. Recuperar documentos (FAISS + Cross-Encoder reranker)
            docs = self.retriever.invoke(question)

            # 2. Montar contexto
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            # 3. Construir prompt com aprendizado injetado
            prompt = self._build_prompt(question, context)

            # 4. Gerar resposta
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content[:200],
                        "source":  doc.metadata.get("source", "Desconhecido"),
                    }
                    for doc in docs
                ],
            }

        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {e}")
            return {
                "answer":  f"Desculpe, ocorreu um erro: {str(e)}",
                "sources": [],
            }

    # ── Utilitários ───────────────────────────────────────────────────────────

    def reload_learning(self) -> None:
        """
        Força o recarregamento do LearningStore.
        Útil para aplicar novo aprendizado sem reiniciar o sistema.
        """
        if self.learning_store:
            logger.info("🔄 Recarregando aprendizado no RAGChain...")
            # LearningStore lê do disco a cada chamada, então basta logar
            summary = self.learning_store.get_summary()
            logger.info(
                f"   Regras: {summary['rules_count']} | "
                f"Exemplos: {summary['examples_count']}"
            )
        else:
            logger.warning("LearningStore não está disponível.")

    def get_learning_status(self) -> dict:
        """Retorna o estado atual do aprendizado injetado nas respostas."""
        if not self.learning_store:
            return {"available": False}

        summary = self.learning_store.get_summary()
        return {
            "available":      True,
            "rules_count":    summary["rules_count"],
            "examples_count": summary["examples_count"],
            "rules":          self.learning_store.load_rules(),
        }


# Teste
if __name__ == "__main__":
    print("✅ LLMChain pronto para usar")