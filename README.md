# 📚 RAG System - Retrieval-Augmented Generation

Sistema de Q&A inteligente que combina busca de documentos com geração de texto usando IA.

## 🎯 Características

- ✅ Carrega múltiplos formatos (PDF, TXT, DOCX, Markdown)
- ✅ Busca inteligente com FAISS vector database
- ✅ Geração de respostas com Ollama (Mistral) - **Gratuito e Local**
- ✅ Interface web com Streamlit
- ✅ Cache de embeddings para performance
- ✅ Histórico de perguntas

## 🚀 Quick Start

### 1. Clone/Setup

```bash
git clone seu-repo
cd rag-system
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

### 2. Instale Ollama

- Acesse: https://ollama.ai
- Baixe e instale
- Execute: `ollama pull mistral`

### 3. Configure

```bash
cp .env.example .env
# .env já vem com valores padrão
```

### 4. Adicione Documentos

Coloque seus PDFs/TXTs em:

data/documents/

### 5. Rode

```bash
streamlit run app.py
```

Abra http://localhost:8501 no navegador.

## 📁 Estrutura do Projeto

rag-system/
├── src/
│   ├── embeddings/          # Gerenciamento de embeddings
│   ├── retrieval/           # Vector store FAISS
│   ├── generation/          # LLM chain
│   ├── rag_system.py        # Sistema principal
│   └── utils.py             # Funções auxiliares
├── data/
│   ├── documents/           # Seus documentos aqui
│   └── embeddings/          # Cache do FAISS
├── config/
│   └── config.yaml          # Configurações
├── app.py                   # Interface Streamlit
├── requirements.txt         # Dependências
└── README.md

## 🛠️ Como Usar

### Na Interface

1. **Inicializar Sistema**: Clique em "🚀 Inicializar Sistema"
2. **Adicionar Documentos**: Coloque arquivos em `data/documents/`
3. **Construir Base**: Clique em "🔨 Construir Base de Conhecimento"
4. **Fazer Perguntas**: Digite suas perguntas!

### Via Python

```python
from src.rag_system import RAGSystem

# Inicializar
rag = RAGSystem()

# Construir base
rag.build_knowledge_base()

# Fazer pergunta
result = rag.query("Sua pergunta aqui?")
print(result["answer"])
print(result["sources"])
```

## 🔧 Configuração Avançada

Edite `config/config.yaml`:

```yaml
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2

generation:
  temperature: 0.7  # 0=determinístico, 1=criativo
  max_tokens: 500

retrieval:
  k: 3  # Número de documentos relevantes
```

## 📊 Formatos Suportados

- 📄 PDF
- 📝 TXT
- 📋 DOCX
- 📖 Markdown (.md)

## 🆓 Por Que Ollama?

- ✅ **Gratuito**: Sem custos de API
- ✅ **Local**: Seus dados ficam na sua máquina
- ✅ **Rápido**: Funciona offline
- ✅ **Privado**: Sem enviar nada para ninguém

## 🚀 Próximos Passos

- [ ] Adicionar suporte a web scraping
- [ ] Implementar feedback loop (avaliar respostas)
- [ ] Adicionar filtros por data
- [ ] Deploy em produção (Docker)

## 📄 Requisitos

- Python 3.12+
- Ollama instalado
- ~500MB de RAM disponível
- Internet (apenas para primeira execução)

## 📝 Licença

MIT

## 👨‍💻 Autor

Fernando - Engenheiro de Computação

## 🤝 Contribuições

Contribuições são bem-vindas! Abra uma issue ou pull request.