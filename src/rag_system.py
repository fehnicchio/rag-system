# src/rag_system.py
import os
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, get_api_key, ensure_directories_exist, list_documents
from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.vector_store import VectorStore
from src.generation.llm_chain import RAGChain
from src.ranking.reranker import DocumentReranker

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Sistema RAG completo.
    Pipeline: FAISS → Cross-Encoder Reranker → LLM (Ollama)

    Suporta:
      - Múltiplos modelos Ollama (troca em runtime)
      - Modos de conhecimento: docs_only | hybrid
      - Aprendizado contínuo via FeedbackLearner
    """

    RETRIEVAL_POOL_SIZE = 8

    def __init__(
        self,
        config_path: str = 'config/config.yaml',
        model_name: str = 'mistral',
        knowledge_mode: str = 'docs_only',
    ):
        logger.info("=" * 50)
        logger.info("Inicializando Sistema RAG")
        logger.info("=" * 50)

        self.config         = load_config(config_path)
        self.model_name     = model_name
        self.knowledge_mode = knowledge_mode

        ensure_directories_exist(self.config)

        self._init_embeddings()
        self._init_vector_store()
        self._init_llm_chain()

        logger.info("✅ Sistema RAG inicializado com sucesso!\n")

    # ─── Inicialização ─────────────────────────────────────────────────────────

    def _init_embeddings(self):
        self.embedding_manager = EmbeddingManager(
            model_name=self.config['embeddings']['model']
        )

    def _init_vector_store(self):
        self.vector_store = VectorStore(
            embeddings=self.embedding_manager.get_embeddings(),
            index_path=self.config['paths']['faiss_index']
        )
        try:
            self.vector_store.load_index()
            self.retriever_ready = True
            logger.info("✅ Índice FAISS carregado")
        except Exception:
            logger.info("ℹ️  Índice não encontrado. Construa a base primeiro.")
            self.retriever_ready = False

    def _init_llm_chain(self):
        try:
            api_key = get_api_key('OPENAI_API_KEY')

            if not self.retriever_ready:
                self.llm_chain = None
                logger.warning("⚠️  LLM Chain não inicializado (índice ausente)")
                return

            self.reranker = DocumentReranker()

            pool_size    = self._calculate_pool_size()
            final_k      = self.config['retrieval']['k']

            logger.info(
                f"📐 Retrieval: FAISS={pool_size} chunks → "
                f"Re-rank → top {final_k}"
            )

            base_retriever = self.vector_store.get_retriever(k=pool_size)
            retriever      = self._create_reranking_retriever(base_retriever, self.reranker, final_k)

            self.llm_chain = RAGChain(
                retriever      = retriever,
                api_key        = api_key,
                temperature    = self.config['generation']['temperature'],
                max_tokens     = self.config['generation']['max_tokens'],
                model_name     = self.model_name,
                knowledge_mode = self.knowledge_mode,
            )

        except Exception as e:
            logger.error(f"Erro ao inicializar LLM: {e}")
            self.llm_chain = None

    def _calculate_pool_size(self) -> int:
        """Calcula pool de documentos para o FAISS (busca ampla antes do rerank)."""
        try:
            total = self.vector_store.vector_store.index.ntotal
            pool  = min(max(self.RETRIEVAL_POOL_SIZE, total), total)
            logger.info(f"  Vetores FAISS: {total} → pool={pool}")
            return pool
        except Exception:
            return self.RETRIEVAL_POOL_SIZE

    def _create_reranking_retriever(self, base_retriever, reranker, top_k: int):
        """Wrapper que aplica Cross-Encoder após busca vetorial."""

        class RerankerRetriever:
            def __init__(self, base_ret, rerank, k):
                self.base_retriever = base_ret
                self.reranker       = rerank
                self.top_k          = k

            def invoke(self, query):
                candidates = self.base_retriever.invoke(query)
                return self.reranker.rerank(query, candidates, top_k=self.top_k)

            def get_relevant_documents(self, query):
                return self.invoke(query)

        return RerankerRetriever(base_retriever, reranker, top_k)

    # ─── Troca de modelo / modo em runtime ────────────────────────────────────

    def change_model(self, new_model: str):
        """Troca o modelo Ollama sem reconstruir o índice FAISS."""
        if not self.llm_chain:
            raise RuntimeError("LLM Chain não inicializado. Construa a base primeiro.")
        self.llm_chain.change_model(new_model)
        self.model_name = new_model
        logger.info(f"✅ Modelo alterado para: {new_model}")

    def change_knowledge_mode(self, new_mode: str):
        """Troca o modo de conhecimento em runtime."""
        if not self.llm_chain:
            raise RuntimeError("LLM Chain não inicializado.")
        self.llm_chain.change_knowledge_mode(new_mode)
        self.knowledge_mode = new_mode
        logger.info(f"✅ Modo alterado para: {new_mode}")

    # ─── API pública ───────────────────────────────────────────────────────────

    def build_knowledge_base(self) -> bool:
        """Constrói o índice FAISS a partir dos documentos em disco."""
        logger.info("Construindo base de conhecimento...")

        doc_paths = list_documents(self.config['paths']['documents'])
        if not doc_paths:
            logger.error("❌ Nenhum documento encontrado")
            return False

        logger.info(f"📄 {len(doc_paths)} documentos encontrados")

        try:
            chunks = self.vector_store.load_documents(
                document_paths = doc_paths,
                chunk_size     = self.config['documents']['chunk_size'],
                chunk_overlap  = self.config['documents']['chunk_overlap']
            )

            if not chunks:
                logger.error("❌ Nenhum chunk gerado")
                return False

            logger.info(f"  Total de chunks: {len(chunks)}")
            self.vector_store.build_index(chunks, save=True)

            self.retriever_ready = True
            self._init_llm_chain()

            logger.info("✅ Base de conhecimento construída!")
            return True

        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False

    def query(self, question: str, knowledge_mode: str = None) -> dict:
        """
        Faz uma pergunta ao sistema RAG.

        Args:
            question:       Pergunta do usuário
            knowledge_mode: Sobrescreve o modo padrão se informado

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        if not self.retriever_ready or not self.llm_chain:
            return {
                "answer":  "❌ Sistema não está pronto. Construa a base de conhecimento primeiro.",
                "sources": [],
            }
        return self.llm_chain.query(question, knowledge_mode=knowledge_mode)

    def get_status(self) -> dict:
        return {
            "embeddings_ready":   True,
            "vector_store_ready": self.retriever_ready,
            "llm_ready":          self.llm_chain is not None,
            "system_ready":       self.retriever_ready and self.llm_chain is not None,
            "model":              self.model_name,
            "knowledge_mode":     self.knowledge_mode,
        }


# ── Teste standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGSystem()
    print("\n📊 Status:")
    print(rag.get_status())