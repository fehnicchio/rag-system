# src/utils.py
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

def load_config(config_path='config/config.yaml'):
    """Carregar configurações do arquivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Config carregada de: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Arquivo de config não encontrado: {config_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao carregar config: {e}")
        raise

def get_api_key(key_name='OPENAI_API_KEY'):
    """Obter chave de API do .env"""
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"⚠️ {key_name} não encontrada em .env")
    logger.info(f"✅ {key_name} carregada com sucesso")
    return api_key

def ensure_directories_exist(config):
    """Garantir que diretórios existem"""
    paths = config.get('paths', {})
    for key, path in paths.items():
        if isinstance(path, str):
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Diretório '{path}' pronto")

def list_documents(documents_path):
    """Listar documentos disponíveis"""
    documents = []
    supported_formats = ('.pdf', '.txt', '.docx', '.md')
    
    if not os.path.exists(documents_path):
        logger.warning(f"Diretório '{documents_path}' não existe")
        return documents
    
    for file in os.listdir(documents_path):
        if file.endswith(supported_formats):
            full_path = os.path.join(documents_path, file)
            documents.append(full_path)
    
    return documents

# Teste rápido
if __name__ == "__main__":
    config = load_config()
    print("✅ Config carregada:", config.keys())
    ensure_directories_exist(config)
    print("✅ Diretórios prontos")