# src/ranking/reranker.py
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DocumentReranker:
    """
    Re-ranking de documentos usando Cross-Encoder.

    Diferença entre Bi-Encoder (abordagem anterior) e Cross-Encoder:
    - Bi-Encoder: embed(query) + embed(doc) → similaridade cosseno
      → rápido, mas query e doc são comparados de forma independente
    - Cross-Encoder: modelo(query + doc) → score de relevância direto
      → mais lento, mas entende a relação entre query e doc,
        muito mais preciso para reranking

    O fluxo correto é:
      1. FAISS (Bi-Encoder, rápido) busca um pool amplo de candidatos
      2. Cross-Encoder (lento, preciso) reordena esse pool
    """

    # Modelo padrão: leve, bom equilíbrio entre velocidade e precisão.
    # Alternativas mais precisas (porém mais lentas):
    #   'cross-encoder/ms-marco-MiniLM-L-12-v2'
    #   'cross-encoder/ms-marco-electra-base'
    DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Inicializar o Cross-Encoder.

        Args:
            model_name: modelo HuggingFace cross-encoder a usar
        """
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Carregando Cross-Encoder: {model_name}")
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            self._use_cross_encoder = True
            logger.info("✅ Cross-Encoder carregado com sucesso")
        except ImportError:
            logger.warning(
                "⚠️  sentence-transformers não instalado. "
                "Instale com: pip install sentence-transformers\n"
                "    Usando fallback por similaridade de embedding (menos preciso)."
            )
            self._use_cross_encoder = False
            self._init_fallback()
        except Exception as e:
            logger.error(f"Erro ao carregar Cross-Encoder ({e}). Usando fallback.")
            self._use_cross_encoder = False
            self._init_fallback()

    def _init_fallback(self):
        """Fallback: bi-encoder com all-MiniLM-L6-v2."""
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Fallback bi-encoder carregado")

    # ─────────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────────

    def rerank(self, query: str, documents: list, top_k: int = 3) -> list:
        """
        Re-rankear documentos pelo score de relevância real.

        Args:
            query:     pergunta do usuário
            documents: lista de LangChain Document objects (candidatos do FAISS)
            top_k:     quantos documentos retornar após o reranking

        Returns:
            Lista de Documents ordenada pelo score (mais relevante primeiro)
        """
        if not documents:
            return []

        scored = self.rerank_with_scores(query, documents, top_k=top_k)
        return [doc for doc, _score in scored]

    def rerank_with_scores(self, query: str, documents: list, top_k: int = 3) -> list:
        """
        Re-rankear e retornar (documento, score).

        Args:
            query, documents, top_k: idem a rerank()

        Returns:
            Lista de tuplas (Document, float) ordenada por score decrescente
        """
        if not documents:
            return []

        logger.info(f"Re-ranking {len(documents)} documentos → top {top_k}...")

        if self._use_cross_encoder:
            scored = self._cross_encoder_scores(query, documents)
        else:
            scored = self._biencoder_scores(query, documents)

        # Ordenar por score decrescente e pegar top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        result = scored[:top_k]

        # Log dos vencedores
        for i, (doc, score) in enumerate(result):
            source = doc.metadata.get('source', 'Desconhecido')
            logger.info(f"  {i+1}. score={score:.4f}  |  {source[:80]}")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Implementações internas
    # ─────────────────────────────────────────────────────────────────────────

    def _cross_encoder_scores(self, query: str, documents: list) -> list:
        """
        Usa Cross-Encoder para pontuar cada par (query, documento).

        O modelo recebe query + texto do documento concatenados e retorna
        diretamente um score de relevância — muito mais preciso que cosine
        similarity entre embeddings independentes.
        """
        try:
            pairs = []
            for doc in documents:
                content = doc.page_content
                # Cross-Encoders têm limite de tokens; representamos documentos
                # longos com início + fim para não perder contexto do final.
                if len(content) > 512:
                    half = 256
                    content = content[:half] + " [...] " + content[-half:]
                pairs.append([query, content])

            scores = self.model.predict(pairs)  # ndarray de floats

            return list(zip(documents, scores.tolist()))

        except Exception as e:
            logger.error(f"Erro no Cross-Encoder: {e}. Usando fallback.")
            return self._biencoder_scores(query, documents)

    def _biencoder_scores(self, query: str, documents: list) -> list:
        """
        Fallback: similaridade cosseno entre embeddings (menos preciso).
        Usado quando sentence-transformers não está disponível.
        """
        query_emb = np.array(self.embeddings.embed_query(query))
        scored = []
        for doc in documents:
            content = doc.page_content
            if len(content) > 1000:
                beginning = content[:400]
                middle = content[len(content) // 2: len(content) // 2 + 400]
                end = content[-200:]
                content = f"{beginning}\n...\n{middle}\n...\n{end}"
            doc_emb = np.array(self.embeddings.embed_query(content))
            score = float(np.dot(query_emb, doc_emb))
            scored.append((doc, score))
        return scored


# ─── Teste standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.documents import Document

    reranker = DocumentReranker()

    test_query = "O que é visão computacional?"
    test_docs = [
        Document(page_content="Python é uma linguagem de programação de alto nível.", metadata={"source": "doc_python.pdf"}),
        Document(page_content="Visão computacional é o campo de IA que ensina máquinas a interpretar imagens e vídeos.", metadata={"source": "doc_visao.pdf"}),
        Document(page_content="Redes neurais convolucionais (CNNs) revolucionaram a visão computacional.", metadata={"source": "doc_cnn.pdf"}),
        Document(page_content="Banco de dados relacionais usam SQL para consultas.", metadata={"source": "doc_sql.pdf"}),
        Document(page_content="YOLO é um algoritmo de detecção de objetos em tempo real amplamente usado em visão computacional.", metadata={"source": "doc_yolo.pdf"}),
        Document(page_content="Processamento de linguagem natural é outra área da IA.", metadata={"source": "doc_nlp.pdf"}),
    ]

    print(f"\nQuery: '{test_query}'")
    print(f"Documentos candidatos: {len(test_docs)}")
    print("\nResultado do reranking (top 3):")

    results = reranker.rerank_with_scores(test_query, test_docs, top_k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {doc.metadata['source']}: {doc.page_content[:60]}...")