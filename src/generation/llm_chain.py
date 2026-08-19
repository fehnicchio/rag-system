# src/generation/llm_chain.py
from langchain_ollama import OllamaLLM
import logging

logger = logging.getLogger(__name__)

# Prompts por modo de conhecimento
PROMPTS = {
    'docs_only': """Você é um assistente especializado em análise de documentos.
Responda APENAS com base no contexto dos documentos fornecidos.
Se a resposta não estiver nos documentos, diga claramente:
"Não encontrei essa informação nos documentos carregados."
Sempre indique a fonte: "[resposta], com base nos documentos"
Responda sempre no mesmo idioma da pergunta.
{rules_block}{examples_block}
CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA:
{question}

RESPOSTA:""",

    'hybrid': """Você é um assistente inteligente especializado em análise de documentos.
Priorize responder com base nos documentos fornecidos e no mesmo idioma da pergunta.
Apenas se a resposta NÃO estiver nos documentos, use seu conhecimento geral.
SEMPRE indique a origem da informação:

Se está nos documentos: "[resposta], com base nos documentos (Fonte: nome)"
Se usou conhecimento geral: "Nos documentos não encontrei, mas geralmente: [resposta]"
{rules_block}{examples_block}
CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA:
{question}

RESPOSTA:""",
}


class RAGChain:
    """
    Cadeia de RAG com suporte a múltiplos modelos Ollama,
    modos de conhecimento e aprendizado contínuo via few-shot.
    """

    MAX_FEW_SHOT_EXAMPLES = 3

    def __init__(
        self,
        retriever,
        api_key=None,
        temperature=0.7,
        max_tokens=500,
        model_name='mistral',
        knowledge_mode='docs_only',
    ):
        """
        Args:
            retriever:       Retriever com re-ranking
            api_key:         Mantido por compatibilidade (não usado com Ollama)
            temperature:     Criatividade do LLM (0–1)
            max_tokens:      Tamanho máximo da resposta
            model_name:      Nome do modelo Ollama
            knowledge_mode:  'docs_only' | 'hybrid'
        """
        self.model_name     = model_name
        self.knowledge_mode = knowledge_mode
        self.retriever      = retriever

        logger.info(f"Inicializando LLM: model={model_name} | mode={knowledge_mode}")
        self._load_llm(model_name, temperature)

        # LearningStore (opcional)
        try:
            from src.learning.learning_store import LearningStore
            self.learning_store = LearningStore()
            logger.info("✅ LearningStore conectado")
        except Exception as e:
            logger.warning(f"⚠️ LearningStore não disponível: {e}")
            self.learning_store = None

        logger.info("✅ RAGChain pronto")

    # ── Carregamento do LLM ───────────────────────────────────────────────────

    def _load_llm(self, model_name: str, temperature: float = 0.7):
        """Carrega (ou recarrega) o LLM Ollama."""
        try:
            self.llm = OllamaLLM(model=model_name, temperature=temperature)
            logger.info(f"✅ LLM Ollama carregado: {model_name}")
        except Exception as e:
            logger.error(f"Erro ao carregar LLM '{model_name}': {e}")
            raise

    def change_model(self, new_model: str, temperature: float = 0.7):
        """Troca o modelo em runtime sem reiniciar todo o sistema."""
        logger.info(f"🔄 Trocando modelo: {self.model_name} → {new_model}")
        self._load_llm(new_model, temperature)
        self.model_name = new_model
        logger.info(f"✅ Modelo alterado para: {new_model}")

    def change_knowledge_mode(self, new_mode: str):
        """Troca o modo de conhecimento em runtime."""
        if new_mode not in PROMPTS:
            raise ValueError(f"Modo inválido: {new_mode}. Use: {list(PROMPTS.keys())}")
        self.knowledge_mode = new_mode
        logger.info(f"✅ Modo de conhecimento alterado para: {new_mode}")

    # ── Blocos de aprendizado ─────────────────────────────────────────────────

    def _build_rules_block(self) -> str:
        if not self.learning_store:
            return ""
        rules = self.learning_store.load_rules()
        if not rules:
            return ""
        rules_text = "\n".join(f"- {r['rule']}" for r in rules)
        return f"\nREGRAS DE COMPORTAMENTO (siga obrigatoriamente):\n{rules_text}\n"

    def _build_examples_block(self) -> str:
        if not self.learning_store:
            return ""
        examples = self.learning_store.load_examples(limit=self.MAX_FEW_SHOT_EXAMPLES)
        if not examples:
            return ""
        block = "\nEXEMPLOS DE RESPOSTAS DE QUALIDADE:\n"
        for i, ex in enumerate(examples, 1):
            block += f"\nExemplo {i}:\n  Pergunta: {ex['question']}\n  Resposta: {ex['good_answer']}\n"
        return block + "\n"

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, knowledge_mode: str = None) -> dict:
        """
        Faz uma pergunta ao RAG.

        Args:
            question:       Pergunta do usuário
            knowledge_mode: Sobrescreve o modo padrão se informado

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        mode = knowledge_mode or self.knowledge_mode

        try:
            logger.info(f"Processando: '{question[:80]}' | modelo={self.model_name} | modo={mode}")

            # 1. Recuperar documentos
            docs = self.retriever.invoke(question)

            # 2. Montar contexto
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            # 3. Selecionar template de prompt
            template = PROMPTS.get(mode, PROMPTS['hybrid'])

            # 4. Preencher prompt
            prompt = template.format(
                rules_block    = self._build_rules_block(),
                examples_block = self._build_examples_block(),
                context        = context,
                question       = question,
            )

            # 5. Gerar resposta
            response = self.llm.invoke(prompt)
            answer   = response.content if hasattr(response, "content") else str(response)

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

    def reload_learning(self):
        """Força recarregamento do LearningStore (log apenas)."""
        if self.learning_store:
            summary = self.learning_store.get_summary()
            logger.info(
                f"🔄 Aprendizado recarregado: "
                f"{summary['rules_count']} regras | {summary['examples_count']} exemplos"
            )

    def get_learning_status(self) -> dict:
        if not self.learning_store:
            return {"available": False}
        summary = self.learning_store.get_summary()
        return {
            "available":      True,
            "rules_count":    summary["rules_count"],
            "examples_count": summary["examples_count"],
            "rules":          self.learning_store.load_rules(),
        }

    def get_status(self) -> dict:
        return {
            "model":          self.model_name,
            "knowledge_mode": self.knowledge_mode,
            "learning":       self.get_learning_status(),
        }


# ── Teste standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("✅ LLMChain pronto para usar")
    print("Modelos disponíveis via Ollama: mistral, llama2, neural-chat, openchat, deepseek-r1")
    print("Modos: docs_only, hybrid")