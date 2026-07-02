# src/retrieval/vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """Gerenciar FAISS vector store"""
    
    def __init__(self, embeddings, index_path='data/embeddings/faiss_index'):
        """
        Inicializar vector store
        
        Args:
            embeddings: Objeto de embeddings
            index_path: Caminho para salvar índice FAISS
        """
        self.embeddings = embeddings
        self.index_path = index_path
        self.vector_store = None
    
    def load_documents(self, document_paths, chunk_size=500, chunk_overlap=50):
        """
        Carregar documentos de vários formatos
        
        Args:
            document_paths: Lista de caminhos de documentos
            chunk_size: Tamanho do chunk
            chunk_overlap: Sobreposição entre chunks
        
        Returns:
            Lista de documentos divididos
        """
        documents = []
        
        for path in document_paths:
            logger.info(f"Carregando: {path}")
            
            try:
                if path.endswith('.pdf'):
                    loader = PDFPlumberLoader(path)
                    documents.extend(loader.load())
                
                elif path.endswith('.txt'):
                    loader = TextLoader(path, encoding='utf-8')
                    documents.extend(loader.load())
                
                elif path.endswith('.md'):
                    loader = TextLoader(path, encoding='utf-8')
                    documents.extend(loader.load())
                
                else:
                    logger.warning(f"Formato não suportado: {path}")
                    continue
                
                logger.info(f"✅ {path} carregado ({len(documents)} docs)")
            
            except Exception as e:
                logger.error(f"Erro ao carregar {path}: {e}")
                continue
        
        if not documents:
            logger.warning("Nenhum documento foi carregado")
            return []
        
        # Split em chunks
        logger.info(f"Dividindo {len(documents)} documentos em chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_docs = splitter.split_documents(documents)
        logger.info(f"✅ {len(chunked_docs)} chunks criados")
        
        return chunked_docs
    
    def build_index(self, documents, save=True):
        """
        Construir índice FAISS
        
        Args:
            documents: Lista de documentos
            save: Se deve salvar índice
        """
        logger.info("Construindo índice FAISS...")
        
        try:
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            logger.info("✅ Índice FAISS construído")
            
            if save:
                self.save_index()
            
            return self.vector_store
        
        except Exception as e:
            logger.error(f"Erro ao construir índice: {e}")
            raise
    
    def save_index(self):
        """Salvar índice FAISS em disco"""
        if not self.vector_store:
            raise ValueError("Índice não construído ainda")
        
        # Criar diretório se não existir
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        
        self.vector_store.save_local(self.index_path)
        logger.info(f"✅ Índice salvo em: {self.index_path}")
    
    def load_index(self):
        """Carregar índice FAISS salvo"""
        try:
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings
            )
            logger.info(f"✅ Índice carregado de: {self.index_path}")
            return self.vector_store
        
        except Exception as e:
            logger.error(f"Erro ao carregar índice: {e}")
            raise
    
    def search(self, query, k=3):
        """
        Buscar documentos relevantes
        
        Args:
            query: Texto da busca
            k: Número de resultados
        
        Returns:
            Lista de (document, score)
        """
        if not self.vector_store:
            raise ValueError("Vector store não inicializado")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, k=3):
        """Retornar retriever para LangChain"""
        if not self.vector_store:
            raise ValueError("Vector store não inicializado")
        
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

# Teste
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.embeddings.embedding_manager import EmbeddingManager
    
    em = EmbeddingManager()
    vs = VectorStore(em.get_embeddings())
    
    print("✅ VectorStore pronto para usar")