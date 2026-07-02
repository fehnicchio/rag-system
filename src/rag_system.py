# src/rag_system.py
import os
import logging
import sys
from pathlib import Path

# Adicionar diretório ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, get_api_key, ensure_directories_exist, list_documents
from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.vector_store import VectorStore
from src.generation.llm_chain import RAGChain

logger = logging.getLogger(__name__)

class RAGSystem:
    """Sistema RAG completo"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Inicializar sistema RAG
        
        Args:
            config_path: Caminho do arquivo de configuração
        """
        logger.info("=" * 50)
        logger.info("Inicializando Sistema RAG")
        logger.info("=" * 50)
        
        # Carregar configuração
        self.config = load_config(config_path)
        ensure_directories_exist(self.config)
        
        # Inicializar componentes
        self._init_embeddings()
        self._init_vector_store()
        self._init_llm_chain()
        
        logger.info("✅ Sistema RAG inicializado com sucesso!\n")
    
    def _init_embeddings(self):
        """Inicializar embeddings"""
        model_name = self.config['embeddings']['model']
        self.embedding_manager = EmbeddingManager(model_name=model_name)
    
    def _init_vector_store(self):
        """Inicializar vector store"""
        index_path = self.config['paths']['faiss_index']
        self.vector_store = VectorStore(
            embeddings=self.embedding_manager.get_embeddings(),
            index_path=index_path
        )
        
        # Tentar carregar índice existente
        try:
            self.vector_store.load_index()
            self.retriever_ready = True
            logger.info("✅ Índice FAISS carregado do cache")
        except:
            logger.info("ℹ️ Nenhum índice encontrado. Você precisa construir um.")
            self.retriever_ready = False
    
    def _init_llm_chain(self):
        """Inicializar LLM chain"""
        try:
            api_key = get_api_key('OPENAI_API_KEY')
            
            if self.retriever_ready:
                retriever = self.vector_store.get_retriever(
                    k=self.config['retrieval']['k']
                )
                
                self.llm_chain = RAGChain(
                    retriever=retriever,
                    api_key=api_key,
                    temperature=self.config['generation']['temperature'],
                    max_tokens=self.config['generation']['max_tokens']
                )
            else:
                self.llm_chain = None
                logger.warning("⚠️ LLM Chain não inicializado (índice não existe)")
        
        except Exception as e:
            logger.error(f"Erro ao inicializar LLM: {e}")
            self.llm_chain = None
    
    def build_knowledge_base(self):
        """
        Construir base de conhecimento a partir dos documentos
        """
        logger.info("Construindo base de conhecimento...")
        
        documents_path = self.config['paths']['documents']
        doc_paths = list_documents(documents_path)
        
        if not doc_paths:
            logger.error(f"❌ Nenhum documento encontrado em {documents_path}")
            return False
        
        logger.info(f"📄 Encontrados {len(doc_paths)} documentos")
        
        try:
            # Carregar e dividir documentos
            chunked_docs = self.vector_store.load_documents(
                document_paths=doc_paths,
                chunk_size=self.config['documents']['chunk_size'],
                chunk_overlap=self.config['documents']['chunk_overlap']
            )
            
            if not chunked_docs:
                logger.error("❌ Nenhum documento foi processado")
                return False
            
            # Construir índice
            self.vector_store.build_index(chunked_docs, save=True)
            
            # Reinicializar LLM chain com novo índice
            self._init_llm_chain()
            self.retriever_ready = True
            
            logger.info("✅ Base de conhecimento construída com sucesso!")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erro ao construir base: {e}")
            return False
    
    def query(self, question):
        """
        Fazer pergunta ao sistema RAG
        
        Args:
            question: Pergunta do usuário
        
        Returns:
            Dict com resposta e fontes
        """
        if not self.retriever_ready or not self.llm_chain:
            return {
                "answer": "❌ Sistema não está pronto. Execute build_knowledge_base() primeiro.",
                "sources": []
            }
        
        return self.llm_chain.query(question)
    
    def get_status(self):
        """Retornar status do sistema"""
        return {
            "embeddings_ready": True,
            "vector_store_ready": self.retriever_ready,
            "llm_ready": self.llm_chain is not None,
            "system_ready": self.retriever_ready and self.llm_chain is not None
        }

# Teste/Uso
if __name__ == "__main__":
    # Inicializar sistema
    rag = RAGSystem()
    
    # Ver status
    print("\n📊 Status do Sistema:")
    print(rag.get_status())
    
    # Para usar: rag.build_knowledge_base()
    # Depois: result = rag.query("Sua pergunta aqui")