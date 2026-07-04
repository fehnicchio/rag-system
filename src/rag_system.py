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
    """Sistema RAG completo com pipeline FAISS → Cross-Encoder → LLM."""

    # Quantos chunks buscar no FAISS antes de rerankar.
    # Deve ser grande o suficiente para cobrir todos os documentos.
    # O reranker reduz esse pool para config['retrieval']['k'] no final.
    RETRIEVAL_POOL_SIZE = 20  # ← ajuste conforme o tamanho do seu corpus

    def __init__(self, config_path='config/config.yaml'):
        logger.info("=" * 50)
        logger.info("Inicializando Sistema RAG")
        logger.info("=" * 50)

        self.config = load_config(config_path)
        ensure_directories_exist(self.config)

        self._init_embeddings()
        self._init_vector_store()
        self._init_llm_chain()

        logger.info("✅ Sistema RAG inicializado com sucesso!\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Inicialização dos componentes
    # ─────────────────────────────────────────────────────────────────────────

    def _init_embeddings(self):
        model_name = self.config['embeddings']['model']
        self.embedding_manager = EmbeddingManager(model_name=model_name)

    def _init_vector_store(self):
        index_path = self.config['paths']['faiss_index']
        self.vector_store = VectorStore(
            embeddings=self.embedding_manager.get_embeddings(),
            index_path=index_path
        )
        try:
            self.vector_store.load_index()
            self.retriever_ready = True
            logger.info("✅ Índice FAISS carregado do cache")
        except Exception:
            logger.info("ℹ️  Nenhum índice encontrado. Construa a base de conhecimento primeiro.")
            self.retriever_ready = False

    def _init_llm_chain(self):
        try:
            api_key = get_api_key('OPENAI_API_KEY')

            if not self.retriever_ready:
                self.llm_chain = None
                logger.warning("⚠️  LLM Chain não inicializado (índice não existe)")
                return

            # Criar re-ranker (Cross-Encoder)
            self.reranker = DocumentReranker()

            # ── Pool size: quantos chunks buscar no FAISS ──────────────────
            # A ideia é buscar chunks suficientes para garantir que todos os
            # documentos tenham ao menos um representante no pool antes do
            # reranking. Usamos o maior valor entre:
            #   • RETRIEVAL_POOL_SIZE (mínimo configurável)
            #   • n_docs * chunks_por_doc estimado
            # mas limitamos ao total de vetores no índice para não pedir mais
            # do que existe.
            final_k = self.config['retrieval']['k']          # ex: 3 (saída final)
            pool_size = self._calculate_pool_size(final_k)

            logger.info(
                f"📐 Retrieval: FAISS busca {pool_size} chunks → "
                f"Cross-Encoder re-ranka → top {final_k} para o LLM"
            )

            base_retriever = self.vector_store.get_retriever(k=pool_size)
            retriever = self._create_reranking_retriever(
                base_retriever, self.reranker, top_k=final_k
            )

            self.llm_chain = RAGChain(
                retriever=retriever,
                api_key=api_key,
                temperature=self.config['generation']['temperature'],
                max_tokens=self.config['generation']['max_tokens']
            )

        except Exception as e:
            logger.error(f"Erro ao inicializar LLM: {e}")
            self.llm_chain = None

    def _calculate_pool_size(self, final_k: int) -> int:
        """
        Calcula quantos chunks buscar no FAISS para garantir cobertura ampla.

        Lógica:
        - Tenta descobrir o total de vetores no índice FAISS.
        - Pool = max(RETRIEVAL_POOL_SIZE, total_vetores) mas nunca mais que
          o total existente (evita erro do FAISS).
        - Se não conseguir descobrir o total, usa RETRIEVAL_POOL_SIZE.
        """
        try:
            # FAISS expõe o número de vetores via .index.ntotal
            total_vectors = self.vector_store.index.ntotal
            # Nunca pedir mais do que existe; nunca menos que o mínimo definido
            pool = min(max(self.RETRIEVAL_POOL_SIZE, total_vectors), total_vectors)
            logger.info(f"  Vetores no índice FAISS: {total_vectors} → pool={pool}")
            return pool
        except AttributeError:
            # VectorStore não expõe .index diretamente — tenta via get_index()
            try:
                total_vectors = self.vector_store.get_index().ntotal
                pool = min(max(self.RETRIEVAL_POOL_SIZE, total_vectors), total_vectors)
                logger.info(f"  Vetores no índice FAISS: {total_vectors} → pool={pool}")
                return pool
            except Exception:
                logger.warning(
                    f"  Não foi possível ler ntotal do índice FAISS. "
                    f"Usando pool_size={self.RETRIEVAL_POOL_SIZE}"
                )
                return self.RETRIEVAL_POOL_SIZE

    # ─────────────────────────────────────────────────────────────────────────
    # Wrapper de retriever com reranking
    # ─────────────────────────────────────────────────────────────────────────

    def _create_reranking_retriever(self, base_retriever, reranker, top_k: int):
        """Wrapper que aplica Cross-Encoder após a busca vetorial."""

        class RerankerRetriever:
            def __init__(self, base_ret, rerank, k):
                self.base_retriever = base_ret
                self.reranker = rerank
                self.top_k = k

            def invoke(self, query):
                # 1. FAISS recupera pool amplo de candidatos
                candidates = self.base_retriever.invoke(query)
                logger.debug(f"  FAISS retornou {len(candidates)} candidatos")

                # 2. Cross-Encoder re-ranka e seleciona os melhores
                reranked = self.reranker.rerank(query, candidates, top_k=self.top_k)
                logger.debug(f"  Reranker selecionou {len(reranked)} documentos finais")
                return reranked

            # Compatibilidade com chains antigas do LangChain
            def get_relevant_documents(self, query):
                return self.invoke(query)

        return RerankerRetriever(base_retriever, reranker, top_k)

    # ─────────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────────

    def build_knowledge_base(self) -> bool:
        """Constrói o índice FAISS a partir dos documentos em disco."""
        logger.info("Construindo base de conhecimento...")

        documents_path = self.config['paths']['documents']
        doc_paths = list_documents(documents_path)

        if not doc_paths:
            logger.error(f"❌ Nenhum documento encontrado em {documents_path}")
            return False

        logger.info(f"📄 Encontrados {len(doc_paths)} documentos")

        try:
            chunked_docs = self.vector_store.load_documents(
                document_paths=doc_paths,
                chunk_size=self.config['documents']['chunk_size'],
                chunk_overlap=self.config['documents']['chunk_overlap']
            )

            if not chunked_docs:
                logger.error("❌ Nenhum chunk gerado. Verifique os documentos.")
                return False

            logger.info(f"  Total de chunks gerados: {len(chunked_docs)}")

            self.vector_store.build_index(chunked_docs, save=True)
            self._init_llm_chain()   # Reinicializa com o novo índice
            self.retriever_ready = True

            logger.info("✅ Base de conhecimento construída com sucesso!")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao construir base: {e}")
            return False

    def query(self, question: str) -> dict:
        """
        Faz uma pergunta ao sistema RAG.

        Returns:
            {"answer": str, "sources": list}
        """
        if not self.retriever_ready or not self.llm_chain:
            return {
                "answer": "❌ Sistema não está pronto. Execute build_knowledge_base() primeiro.",
                "sources": []
            }
        return self.llm_chain.query(question)

    def get_status(self) -> dict:
        return {
            "embeddings_ready": True,
            "vector_store_ready": self.retriever_ready,
            "llm_ready": self.llm_chain is not None,
            "system_ready": self.retriever_ready and self.llm_chain is not None,
        }


# ─── Teste standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGSystem()

    print("\n📊 Status do Sistema:")
    print(rag.get_status())

    # rag.build_knowledge_base()
    # result = rag.query("Sua pergunta aqui")
    # print(result)