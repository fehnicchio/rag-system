# src/embeddings/embedding_manager.py
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Gerenciar modelo de embeddings"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Inicializar embedding model
        
        Args:
            model_name: Nome do modelo HuggingFace
        """
        logger.info(f"Carregando modelo de embeddings: {model_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embedding model carregado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar embedding model: {e}")
            raise
    
    def get_embeddings(self):
        """Retornar objeto embeddings"""
        return self.embeddings
    
    def embed_text(self, text):
        """Embeddar texto individual"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Erro ao embeddar texto: {e}")
            return None

# Teste
if __name__ == "__main__":
    em = EmbeddingManager()
    test_text = "Isso é um teste de embedding"
    embedding = em.embed_text(test_text)
    print(f"✅ Embedding criado com dimensão: {len(embedding)}")