# app.py
import streamlit as st
import os
import sys
from pathlib import Path

# Adicionar diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem

st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("# 📚 RAG Q&A System")
st.markdown("*Análise inteligente de documentos com IA*")
st.divider()

# Inicializar session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.chat_history = []

# Sidebar - Controles
with st.sidebar:
    st.header("⚙️ Configuração")
    
    # Status
    status = st.session_state.rag_system.get_status() if st.session_state.rag_system else None
    
    if status and status['system_ready']:
        st.markdown(
            '<div class="status-ready">✅ Sistema Pronto</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-error">❌ Sistema não inicializado</div>',
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Inicializar sistema
    if st.button("🚀 Inicializar Sistema", use_container_width=True):
        try:
            with st.spinner("Inicializando..."):
                st.session_state.rag_system = RAGSystem()
            st.success("✅ Sistema inicializado!")
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
    
    st.divider()
    
    # Construir base de conhecimento
    st.subheader("📄 Base de Conhecimento")
    
    # Listar documentos
    from src.utils import list_documents
    doc_path = 'data/documents'
    docs = list_documents(doc_path)
    
    if docs:
        st.info(f"📦 {len(docs)} documento(s) encontrado(s)")
        for doc in docs:
            st.caption(f"• {os.path.basename(doc)}")
    else:
        st.warning("Nenhum documento encontrado em data/documents/")
    
    if st.button("🔨 Construir Base de Conhecimento", use_container_width=True):
        if not st.session_state.rag_system:
            st.error("Primeiro inicialize o sistema!")
        else:
            try:
                with st.spinner("Construindo base..."):
                    success = st.session_state.rag_system.build_knowledge_base()
                
                if success:
                    st.success("✅ Base construída!")
                    st.rerun()
                else:
                    st.error("❌ Erro ao construir base")
            
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
    
    st.divider()
    
    # Histórico
    if st.button("🗑️ Limpar Histórico", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Histórico limpo!")

# Main area
st.subheader("❓ Faça uma Pergunta")

if not st.session_state.rag_system or not st.session_state.rag_system.get_status()['system_ready']:
    st.warning("👈 Complete a configuração na barra lateral")
else:
    # Input
    col1, col2 = st.columns([6, 1])
    
    with col1:
        question = st.text_input(
            "Digite sua pergunta:",
            placeholder="Ex: Qual é o tema principal do documento?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍", use_container_width=True, help="Buscar")
    
    if search_button and question:
        with st.spinner("Buscando resposta..."):
            result = st.session_state.rag_system.query(question)
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"]
            })
        
        # Exibir resultado
        st.divider()
        
        st.subheader("📋 Resposta:")
        st.write(result["answer"])
        
        # Fontes
        if result["sources"]:
            st.subheader("📖 Documentos Relevantes:")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"📄 Fonte {i}: {source['source']}"):
                    st.text(source['content'][:500] + "...")

# Histórico
if st.session_state.chat_history:
    st.divider()
    st.subheader("📝 Histórico de Perguntas")
    
    for i, item in enumerate(st.session_state.chat_history[-5:], 1):
        with st.expander(f"**{i}. {item['question'][:50]}...**"):
            st.write("**Pergunta:**", item['question'])
            st.write("**Resposta:**", item['answer'])