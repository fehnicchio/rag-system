# src/cache/cache_manager.py
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Gerenciar cache de respostas RAG"""
    
    def __init__(self, cache_db='data/response_cache.db'):
        """
        Inicializar cache
        
        Args:
            cache_db: Caminho do banco de cache
        """
        self.cache_db = cache_db
        Path(cache_db).parent.mkdir(parents=True, exist_ok=True)
        self.init_cache()
    
    def init_cache(self):
        """Criar tabela de cache"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT UNIQUE NOT NULL,
                question_text TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                ttl_hours INTEGER DEFAULT 24
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _hash_question(self, question):
        """
        Criar hash da pergunta para busca rápida
        
        Normaliza pergunta (lowercase, sem espaços extras)
        """
        # Normalizar pergunta
        normalized = question.lower().strip()
        # Remover pontuação extra
        normalized = normalized.replace('?', '').replace('!', '')
        
        # Hash SHA256
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def get(self, question):
        """
        Buscar resposta em cache
        
        Args:
            question: Pergunta do usuário
        
        Returns:
            Dict com resposta e metadados, ou None
        """
        try:
            question_hash = self._hash_question(question)
            
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, answer, sources, hit_count, created_at
                FROM response_cache
                WHERE question_hash = ?
            """, (question_hash,))
            
            result = cursor.fetchone()
            
            if result:
                cache_id, answer, sources_json, hit_count, created_at = result
                
                # Atualizar último acesso e contador
                cursor.execute("""
                    UPDATE response_cache
                    SET accessed_at = CURRENT_TIMESTAMP, hit_count = hit_count + 1
                    WHERE id = ?
                """, (cache_id,))
                
                conn.commit()
                
                sources = json.loads(sources_json) if sources_json else []
                
                logger.info(f"✅ Cache HIT: '{question[:50]}...' (acesso #{hit_count + 1})")
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'from_cache': True,
                    'cached_at': created_at
                }
            
            logger.info(f"⏭️ Cache MISS: '{question[:50]}...'")
            conn.close()
            return None
        
        except Exception as e:
            logger.error(f"Erro ao buscar cache: {e}")
            return None
    
    def set(self, question, answer, sources, ttl_hours=24):
        """
        Salvar resposta em cache
        
        Args:
            question: Pergunta original
            answer: Resposta do RAG
            sources: Lista de documentos fonte
            ttl_hours: Tempo de vida em horas
        """
        try:
            question_hash = self._hash_question(question)
            sources_json = json.dumps(sources)
            
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            # Usar INSERT OR REPLACE para atualizar se já existe
            cursor.execute("""
                INSERT OR REPLACE INTO response_cache
                (question_hash, question_text, answer, sources, ttl_hours)
                VALUES (?, ?, ?, ?, ?)
            """, (question_hash, question, answer, sources_json, ttl_hours))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Cache SAVE: '{question[:50]}...'")
        
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
    
    def clear(self):
        """Limpar todo o cache"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM response_cache")
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Cache limpo")
        
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")
    
    def get_statistics(self):
        """Retornar estatísticas do cache"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            # Total de respostas em cache
            cursor.execute("SELECT COUNT(*) FROM response_cache")
            total = cursor.fetchone()[0]
            
            # Total de acessos ao cache
            cursor.execute("SELECT SUM(hit_count) FROM response_cache")
            total_hits = cursor.fetchone()[0] or 0
            
            # Taxa de acerto
            hit_rate = (total_hits / (total + total_hits) * 100) if (total + total_hits) > 0 else 0
            
            conn.close()
            
            return {
                'cached_responses': total,
                'total_cache_hits': total_hits,
                'hit_rate': hit_rate
            }
        
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {
                'cached_responses': 0,
                'total_cache_hits': 0,
                'hit_rate': 0
            }

# Teste
if __name__ == "__main__":
    cache = CacheManager()
    
    # Teste 1: Salvar resposta
    cache.set(
        "O que é RAG?",
        "RAG é uma técnica que combina busca e geração...",
        [{"source": "doc1.pdf"}]
    )
    print("✅ Resposta salva em cache")
    
    # Teste 2: Buscar resposta
    result = cache.get("O que é RAG?")
    if result:
        print(f"✅ Resposta encontrada em cache: {result['answer'][:50]}...")
    
    # Teste 3: Estatísticas
    stats = cache.get_statistics()
    print(f"✅ Cache stats: {stats}")