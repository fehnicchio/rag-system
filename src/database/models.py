# src/database/models.py
import sqlite3
from datetime import datetime
from pathlib import Path
import json

class RAGDatabase:
    """Gerenciar banco de dados SQLite do RAG"""
    
    def __init__(self, db_path='data/rag_feedback.db'):
        """
        Inicializar banco de dados
        
        Args:
            db_path: Caminho do arquivo SQLite
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Criar tabelas se não existirem"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de perguntas/respostas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT DEFAULT 'mistral'
            )
        """)
        
        # Tabela de feedback (👍👎)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                is_helpful BOOLEAN,
                comment TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            )
        """)
        
        # Tabela de estatísticas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_interactions INTEGER DEFAULT 0,
                helpful_count INTEGER DEFAULT 0,
                not_helpful_count INTEGER DEFAULT 0,
                average_response_time REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_interaction(self, question, answer, sources, model_used='mistral'):
        """
        Salvar pergunta e resposta
        
        Returns:
            interaction_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sources_json = json.dumps(sources) if sources else None
        
        cursor.execute("""
            INSERT INTO interactions (question, answer, sources, model_used)
            VALUES (?, ?, ?, ?)
        """, (question, answer, sources_json, model_used))
        
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return interaction_id
    
    def save_feedback(self, interaction_id, is_helpful, comment=''):
        """
        Salvar feedback do usuário
        
        Args:
            interaction_id: ID da interação
            is_helpful: True (👍) ou False (👎)
            comment: Comentário opcional
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (interaction_id, is_helpful, comment)
            VALUES (?, ?, ?)
        """, (interaction_id, is_helpful, comment))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self):
        """Retornar estatísticas gerais"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_helpful = 1")
        helpful = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_helpful = 0")
        not_helpful = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_interactions': total,
            'helpful_count': helpful,
            'not_helpful_count': not_helpful,
            'helpful_rate': (helpful / (helpful + not_helpful) * 100) if (helpful + not_helpful) > 0 else 0
        }
    
    def get_recent_interactions(self, limit=10):
        """Retornar últimas interações"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, question, answer, timestamp
            FROM interactions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results

    def get_negative_feedback(self, limit=50) -> list[dict]:
        """
        Retorna feedbacks negativos com comentário.
        Usado pelo FeedbackLearner e pelo painel de análise no app.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                f.id          AS feedback_id,
                f.comment,
                f.timestamp   AS feedback_ts,
                i.id          AS interaction_id,
                i.question,
                i.answer
            FROM feedback f
            JOIN interactions i ON i.id = f.interaction_id
            WHERE f.is_helpful = 0
              AND f.comment IS NOT NULL
              AND TRIM(f.comment) != ''
            ORDER BY f.timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_all_feedback(self, limit=100) -> list[dict]:
        """Retorna todos os feedbacks (positivos e negativos) com dados da interação."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                f.id          AS feedback_id,
                f.is_helpful,
                f.comment,
                f.timestamp,
                i.question,
                i.answer
            FROM feedback f
            JOIN interactions i ON i.id = f.interaction_id
            ORDER BY f.timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_unprocessed_feedback_count(self) -> int:
        """Retorna quantos feedbacks negativos ainda não foram processados pelo learner."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela pode não existir ainda — criar silenciosamente
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_processed (
                feedback_id INTEGER PRIMARY KEY,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            SELECT COUNT(*) FROM feedback f
            LEFT JOIN feedback_processed fp ON fp.feedback_id = f.id
            WHERE f.is_helpful = 0
              AND f.comment IS NOT NULL
              AND TRIM(f.comment) != ''
              AND fp.feedback_id IS NULL
        """)
        count = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return count


# Teste
if __name__ == "__main__":
    db = RAGDatabase()
    print("✅ Banco de dados criado com sucesso")