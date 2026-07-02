# src/generation/llm_chain.py
from langchain_ollama import OllamaLLM
import logging

logger = logging.getLogger(__name__)

class RAGChain:
    """Gerenciar cadeia de RAG (Retrieval + Generation)"""
    
    def __init__(self, retriever, api_key, temperature=0.7, max_tokens=500):
        """
        Inicializar RAG chain
        
        Args:
            retriever: Retriever do FAISS
            api_key: OpenAI API key
            temperature: Criatividade (0-1)
            max_tokens: Tamanho máximo da resposta
        """
        logger.info("Inicializando LLM...")
        
        try:
            self.llm = OllamaLLM(
                model='mistral',
                temperature=temperature
            )
            logger.info("✅ LLM Ollama carregado (Mistral)")
        except Exception as e:
            logger.error(f"Erro ao carregar LLM: {e}")
            raise
        
        self.retriever = retriever
        logger.info("✅ RAG Chain pronto")
    
    def query(self, question):
        """
        Fazer pergunta ao RAG
        
        Args:
            question: Pergunta do usuário
        
        Returns:
            Dict com answer e source_documents
        """
        try:
            logger.info(f"Processando pergunta: {question}")
            
            # Recuperar documentos relevantes
            docs = self.retriever.invoke(question)
            
            # Criar contexto
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Criar prompt
            prompt = f"""Você é um assistente inteligente especializado em análise de documentos.
            Priorize responder com base nos documentos fornecidos.
            Apenas se a resposta NÃO estiver nos documentos, use seu conhecimento.
            SEMPRE indique a fonte!

            Se a resposta está no documento: 
            "Segundo os documentos: [resposta] (Fonte: nome_do_doc)"

            Se usou conhecimento geral:
            "Nos documentos não encontrei, mas geralmente: [resposta]"

            CONTEXTO:
            {context}

            PERGUNTA:
            {question}

            RESPOSTA:"""
            
            # Gerar resposta
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content if hasattr(response, 'content') else str(response),
                "sources": [
                    {
                        "content": doc.page_content[:200],
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc in docs
                ]
            }
        
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {e}")
            return {
                "answer": f"Desculpe, ocorreu um erro: {str(e)}",
                "sources": []
            }

# Teste
if __name__ == "__main__":
    print("✅ LLMChain pronto para usar")