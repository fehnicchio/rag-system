# app.py
import streamlit as st
import os
import sys
import time
from pathlib import Path

# Adicionar diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem
from src.database.models import RAGDatabase
from src.utils import list_documents

# Configurar página
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
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
    st.session_state.db = RAGDatabase()
    st.session_state.last_interaction_id = None

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
    
    doc_path = 'data/documents'
    docs = list_documents(doc_path)
    
    if docs:
        st.info(f"📦 {len(docs)} documento(s) encontrado(s)")
        with st.expander("📋 Ver documentos"):
            for doc in docs:
                st.caption(f"• {os.path.basename(doc)}")
    else:
        st.warning("Nenhum documento encontrado em data/documents/")
    
    # Abas para diferentes formas de adicionar documentos
    tab1, tab2 = st.tabs(["📁 Arquivos Locais", "🌐 Web Scraping"])
    
    with tab1:
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
    
    with tab2:
        st.subheader("🌐 Adicionar Documentos da Web")
        
        # Inicializar session state para URL
        if 'url_input_key' not in st.session_state:
            st.session_state.url_input_key = 0
        
        url_input = st.text_input(
            "Cole uma URL:",
            placeholder="https://exemplo.com/artigo",
            label_visibility="collapsed",
            key=f"url_input_{st.session_state.url_input_key}"
        )
        
        if st.button("🔗 Scraping + Adicionar", use_container_width=True):
            if not url_input:
                st.error("❌ Digite uma URL válida")
            else:
                try:
                    from src.scraper.web_scraper import WebScraper
                    
                    with st.spinner("Fazendo scraping..."):
                        scraper = WebScraper()
                        scraped_data = scraper.scrape_url(url_input)
                    
                    if scraped_data:
                        # Salvar documento
                        filepath = scraper.save_to_document(scraped_data)
                        st.success(f"✅ Documento salvo: {os.path.basename(filepath)}")
                        
                        # Limpar input da URL
                        st.session_state.url_input_key += 1
                        
                        # Oferecer reconstruir base
                        if st.button("🔨 Reconstruir Base com Novo Documento", use_container_width=True):
                            with st.spinner("Reconstruindo base..."):
                                st.session_state.rag_system.build_knowledge_base()
                            st.success("✅ Base reconstruída com novo documento!")
                            st.rerun()
                    else:
                        st.error("❌ Falha ao fazer scraping. Verifique a URL.")
                
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")
        
        st.divider()
        st.caption("💡 **Dicas para URLs:**")
        st.caption("• Cole qualquer URL (artigos, blogs, documentação)")
        st.caption("• O texto será extraído automaticamente")
        st.caption("• Clique em 'Reconstruir Base' para usar o novo documento")
    
    st.divider()
    st.subheader("📤 Upload de Arquivos Locais")
    
    uploaded_files = st.file_uploader(
        "Selecione arquivos (PDF, TXT, DOCX, MD):",
        type=['pdf', 'txt', 'docx', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("💾 Salvar Arquivos", use_container_width=True):
            try:
                saved_count = 0
                
                for uploaded_file in uploaded_files:
                    # Salvar arquivo em data/documents/
                    doc_path = Path('data/documents') / uploaded_file.name
                    doc_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(doc_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    saved_count += 1
                    st.success(f"✅ {uploaded_file.name} salvo!")
                
                st.divider()
                    
                # Oferecer reconstruir base
                if st.button(f"🔨 Reconstruir Base ({saved_count} novo(s) documento(s))", use_container_width=True):
                    with st.spinner("Reconstruindo base..."):
                        st.session_state.rag_system.build_knowledge_base()
                    st.success(f"✅ Base reconstruída com {saved_count} novo(s) documento(s)!")
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Erro ao salvar: {str(e)}")
            
        st.divider()
        st.caption("💡 **Dicas para Upload:**")
        st.caption("• Selecione múltiplos arquivos de uma vez")
        st.caption("• Suporta PDF, TXT, DOCX e Markdown")
        st.caption("• Clique em 'Reconstruir Base' para usar os novos documentos")
    
    # Estatísticas
    st.subheader("📊 Estatísticas")
    
    # Forçar leitura nova do banco (sem cache)
    if 'stats_placeholder' not in st.session_state:
        st.session_state.stats_placeholder = st.empty()
    
    with st.session_state.stats_placeholder.container():
        # Criar nova conexão para garantir dados atuais
        import sqlite3
        from src.database.models import RAGDatabase
        db_temp = RAGDatabase()
        stats = db_temp.get_statistics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Perguntas", stats['total_interactions'])
    with col2:
        st.metric("Taxa de Utilidade", f"{stats['helpful_rate']:.1f}%")
    
    st.metric("Respostas Úteis", f"{stats['helpful_count']} 👍")
    st.metric("Respostas Ruins", f"{stats['not_helpful_count']} 👎")
    
    st.divider()
    
    # Histórico
    if st.button("🗑️ Limpar Histórico da Sessão", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Histórico limpo!")

# Main area
st.subheader("❓ Faça uma Pergunta")

if not st.session_state.rag_system or not st.session_state.rag_system.get_status()['system_ready']:
    st.warning("👈 Complete a configuração na barra lateral")
else:
    # Input com Form (captura Enter)
    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            question = st.text_input(
                "Digite sua pergunta:",
                placeholder="Ex: Qual é o tema principal do documento?",
                label_visibility="collapsed"
            )
        
        with col2:
            search_button = st.form_submit_button("🔍", use_container_width=True, help="Buscar")
    
    if search_button and question:
        with st.spinner("Buscando resposta..."):
            start_time = time.time()
            result = st.session_state.rag_system.query(question)
            response_time = time.time() - start_time
            
            # Salvar no banco de dados
            interaction_id = st.session_state.db.save_interaction(
                question=question,
                answer=result["answer"],
                sources=result["sources"],
                model_used='mistral'
            )
            st.session_state.last_interaction_id = interaction_id
            
            # Guardar em session state para feedback
            st.session_state.current_interaction_id = interaction_id
            
            # Adicionar ao histórico
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "interaction_id": interaction_id,
                "response_time": response_time
            })
        
        # Exibir resultado
        st.divider()
        
        st.subheader("📋 Resposta:")
        st.write(result["answer"])
        
        # Tempo de resposta
        st.caption(f"⏱️ Tempo de resposta: {response_time:.2f}s")
        
        # Feedback
        st.subheader("⭐ Sua Opinião")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👍 Útil", use_container_width=True, key=f"helpful_{st.session_state.current_interaction_id}"):
                st.session_state.db.save_feedback(
                    interaction_id=st.session_state.current_interaction_id,
                    is_helpful=True
                )
                st.success("✅ Obrigado! Marcado como útil")
                # Pequeno delay para garantir que salvou
                import time
                time.sleep(0.5)
                st.rerun()
        
        with col2:
            if st.button("👎 Não Útil", use_container_width=True, key=f"not_helpful_{st.session_state.current_interaction_id}"):
                st.session_state.db.save_feedback(
                    interaction_id=st.session_state.current_interaction_id,
                    is_helpful=False
                )
                st.success("✅ Obrigado pelo feedback!")
                # Pequeno delay para garantir que salvou
                import time
                time.sleep(0.5)
                st.rerun()
        
        with col3:
            comment = st.text_input(
                "Deixe um comentário (opcional):",
                label_visibility="collapsed",
                placeholder="O que melhorar?",
                key=f"comment_{st.session_state.current_interaction_id}"
            )
            if comment and st.button("💬 Enviar Comentário", use_container_width=True, key=f"comment_btn_{st.session_state.current_interaction_id}"):
                st.session_state.db.save_feedback(
                    interaction_id=st.session_state.current_interaction_id,
                    is_helpful=None,
                    comment=comment
                )
                st.success("✅ Comentário salvo!")
                # Pequeno delay para garantir que salvou
                import time
                time.sleep(0.5)
                st.rerun()
        
        # Fontes
        if result["sources"]:
            st.subheader("📖 Documentos Relevantes:")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"📄 Fonte {i}: {source['source']}"):
                    st.text(source['content'][:500] + "...")

# Histórico da sessão
if st.session_state.chat_history:
    st.divider()
    st.subheader("📝 Histórico de Perguntas (Sessão Atual)")
    
    for i, item in enumerate(st.session_state.chat_history[-5:], 1):
        with st.expander(f"**{i}. {item['question'][:50]}...** ({item['response_time']:.2f}s)"):
            st.write("**Pergunta:**", item['question'])
            st.write("**Resposta:**", item['answer'])
            if item['sources']:
                st.write("**Fontes:**", len(item['sources']), "documento(s)")