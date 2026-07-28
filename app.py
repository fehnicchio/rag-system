# app.py
import streamlit as st
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem
from src.database.models import RAGDatabase
from src.cache.cache_manager import CacheManager
from src.utils import list_documents

# ─── Configurar página ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .rag-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
        color: white;
    }
    .rag-header h1 { margin: 0 0 4px 0; font-size: 1.8rem; font-weight: 700; }
    .rag-header p  { margin: 0; opacity: 0.8; font-size: 0.95rem; }

    .status-card {
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
        font-size: 0.9rem; font-weight: 500;
    }
    .status-ready { background: #ecfdf5; color: #065f46; border: 1px solid #6ee7b7; }
    .status-error { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
    .status-warn  { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
    .status-info  { background: #eff6ff; color: #1e40af; border: 1px solid #93c5fd; }

    .answer-card {
        background: #f8faff; border: 1px solid #dbeafe;
        border-left: 4px solid #3b82f6; border-radius: 10px;
        padding: 20px 24px; line-height: 1.7; font-size: 0.97rem;
        color: #1e293b; margin: 12px 0;
    }

    .time-badge {
        display: inline-block; background: #f1f5f9; color: #64748b;
        font-size: 0.78rem; padding: 3px 10px; border-radius: 20px;
        margin-top: 6px; font-weight: 500;
    }
    .cache-badge { background: #fef9c3; color: #854d0e; }

    /* ── Painel inferior (toolbar) ── */
    .bottom-toolbar {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: white; border-top: 1px solid #e2e8f0;
        z-index: 999; padding: 0;
    }

    /* ── Painel expansível ── */
    .panel-expanded {
        background: white; border-top: 1px solid #e2e8f0;
        padding: 20px 32px; max-height: 420px; overflow-y: auto;
    }

    [data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 10px 14px;
    }

    .model-option {
        border: 2px solid #e2e8f0; border-radius: 10px;
        padding: 14px 16px; cursor: pointer; transition: all 0.2s;
        margin-bottom: 8px;
    }
    .model-option:hover { border-color: #3b82f6; background: #eff6ff; }
    .model-selected { border-color: #3b82f6 !important; background: #eff6ff !important; }
    .model-name { font-weight: 600; font-size: 0.95rem; color: #1e3a5f; }
    .model-desc { font-size: 0.82rem; color: #64748b; margin-top: 2px; }

    .stTextInput > label[data-testid="stWidgetLabel"] { display: none; }

    /* Espaço no final para o toolbar não cobrir conteúdo */
    .main-content-padding { padding-bottom: 120px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>📚 RAG Q&amp;A System</h1>
    <p>Análise inteligente de documentos com Inteligência Artificial</p>
</div>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if 'initialized' not in st.session_state:
    st.session_state.initialized        = True
    st.session_state.rag_system         = None
    st.session_state.chat_history       = []
    st.session_state.db                 = RAGDatabase()
    st.session_state.cache              = CacheManager()
    st.session_state.last_result        = None
    st.session_state.last_interaction_id = None
    st.session_state.feedback_submitted = set()
    st.session_state.feedback_ratings   = {}
    st.session_state.url_input_key      = 0
    # Painel inferior
    st.session_state.panel_open         = None   # None | 'settings' | 'stats'
    # Configurações do modelo
    st.session_state.selected_model     = 'mistral'
    st.session_state.knowledge_mode     = 'docs_only'  # 'docs_only' | 'hybrid'

# ─── Modelos disponíveis ──────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    'mistral': {
        'label': 'Mistral 7B',
        'icon':  '⚡',
        'desc':  'Rápido e eficiente. Ótimo para perguntas diretas.',
        'speed': 'Rápido',
    },
    'llama2': {
        'label': 'Llama 2 7B',
        'icon':  '🦙',
        'desc':  'Meta AI. Boa cobertura geral e raciocínio.',
        'speed': 'Médio',
    },
    'neural-chat': {
        'label': 'Neural Chat',
        'icon':  '🧠',
        'desc':  'Otimizado para conversas e explicações detalhadas.',
        'speed': 'Médio',
    },
    'openchat': {
        'label': 'OpenChat 3.5',
        'icon':  '💬',
        'desc':  'Leve e rápido. Bom para respostas concisas.',
        'speed': 'Muito rápido',
    },
    'deepseek-r1': {
        'label': 'DeepSeek R1',
        'icon':  '🔍',
        'desc':  'Excelente raciocínio e análise de documentos.',
        'speed': 'Lento',
    },
}

KNOWLEDGE_MODES = {
    'docs_only': {
        'label': '📄 Apenas documentos',
        'desc':  'Responde somente com base nos documentos carregados. Mais preciso e confiável.',
        'icon':  '📄',
    },
    'hybrid': {
        'label': '🌐 Documentos + Conhecimento geral',
        'desc':  'Usa documentos como prioridade e complementa com conhecimento geral da IA.',
        'icon':  '🌐',
    },
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuração")

    status = st.session_state.rag_system.get_status() if st.session_state.rag_system else None

    if status and status['system_ready']:
        st.markdown('<div class="status-card status-ready">✅ Sistema pronto e operacional</div>', unsafe_allow_html=True)
    elif status:
        st.markdown('<div class="status-card status-warn">⚠️ Sistema iniciado, mas índice não construído</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card status-error">❌ Sistema não inicializado</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("🚀 Inicializar Sistema", use_container_width=True, type="primary"):
        try:
            with st.spinner("Carregando modelos e índices..."):
                st.session_state.rag_system = RAGSystem(
                    model_name=st.session_state.selected_model
                )
            st.success("✅ Sistema inicializado!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")

    st.divider()

    # ── Base de conhecimento ──
    st.markdown("#### 📄 Base de Conhecimento")
    docs = list_documents('data/documents')

    if docs:
        st.info(f"📦 {len(docs)} documento(s) encontrado(s)")
        with st.expander("📋 Ver documentos"):
            for doc in docs:
                st.caption(f"• {os.path.basename(doc)}")
    else:
        st.warning("Nenhum documento em `data/documents/`")

    tab1, tab2 = st.tabs(["📁 Arquivos", "🌐 Web"])

    with tab1:
        if st.button("🔨 Construir Base", use_container_width=True):
            if not st.session_state.rag_system:
                st.error("Inicialize o sistema primeiro!")
            else:
                with st.spinner("Processando documentos..."):
                    success = st.session_state.rag_system.build_knowledge_base()
                if success:
                    st.success("✅ Base construída!")
                    st.rerun()
                else:
                    st.error("❌ Falha ao construir a base.")

        uploaded_files = st.file_uploader(
            "PDF, TXT, DOCX, MD",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            if st.button(f"💾 Salvar {len(uploaded_files)} arquivo(s)", use_container_width=True):
                for f in uploaded_files:
                    dest = Path('data/documents') / f.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, 'wb') as fp:
                        fp.write(f.getbuffer())
                st.success(f"✅ {len(uploaded_files)} arquivo(s) salvo(s)! Reconstrua a base.")

    with tab2:
        url_input = st.text_input(
            "URL:",
            placeholder="https://exemplo.com/artigo",
            label_visibility="collapsed",
            key=f"url_input_{st.session_state.url_input_key}"
        )
        if st.button("🔗 Fazer Scraping", use_container_width=True):
            if not url_input:
                st.error("❌ Digite uma URL válida")
            elif not st.session_state.rag_system:
                st.error("❌ Inicialize o sistema primeiro")
            else:
                try:
                    from src.scraper.web_scraper import WebScraper
                    with st.spinner("Extraindo conteúdo..."):
                        scraper = WebScraper()
                        data = scraper.scrape_url(url_input)
                    if data:
                        fp = scraper.save_to_document(data)
                        st.session_state.url_input_key += 1
                        st.success(f"✅ Salvo: `{os.path.basename(fp)}`")
                        st.info("Reconstrua a base para incluir este documento.")
                    else:
                        st.error("❌ Falha ao extrair conteúdo.")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

    st.divider()

    # ── Aprendizado ──
    st.markdown("#### 🧠 Aprendizado Contínuo")
    try:
        from src.learning.learning_store import LearningStore
        _store   = LearningStore()
        _summary = _store.get_summary()
        _pending = st.session_state.db.get_unprocessed_feedback_count() if hasattr(st.session_state.db, 'get_unprocessed_feedback_count') else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Regras", _summary["rules_count"])
        with col2:
            st.metric("Exemplos", _summary["examples_count"])

        if _pending > 0:
            st.warning(f"⏳ {_pending} feedback(s) pendente(s)")
    except Exception:
        st.caption("Aprendizado não inicializado")
        _pending = 0

    if st.button("🔍 Analisar e Aprender", use_container_width=True, type="primary"):
        try:
            from src.learning.feedback_learner import FeedbackLearner
            with st.spinner("Processando feedbacks..."):
                learner = FeedbackLearner()
                result  = learner.analyze_feedback()
            if result["processed"] == 0:
                st.info("ℹ️ Nenhum feedback novo.")
            else:
                st.success(
                    f"✅ {result['processed']} feedback(s)  |  "
                    f"📌 {len(result['new_rules'])} regra(s)  |  "
                    f"✍️ {result['new_examples']} exemplo(s)"
                )
                if st.session_state.rag_system and st.session_state.rag_system.llm_chain:
                    st.session_state.rag_system.llm_chain.reload_learning()
                st.rerun()
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")

    st.divider()

    if st.button("🗑️ Limpar Histórico da Sessão", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.session_state.last_interaction_id = None
        st.success("Histórico limpo!")

# ─── Área principal ───────────────────────────────────────────────────────────
st.markdown('<div class="main-content-padding">', unsafe_allow_html=True)

if not st.session_state.rag_system or not st.session_state.rag_system.get_status()['system_ready']:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
        <div style="font-size: 3rem; margin-bottom: 12px;">👈</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #475569;">Complete a configuração na barra lateral</div>
        <div style="font-size: 0.9rem; margin-top: 8px;">Inicialize o sistema e construa a base de conhecimento para começar</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Indicadores ativos ──
    model_info = AVAILABLE_MODELS.get(st.session_state.selected_model, AVAILABLE_MODELS['mistral'])
    mode_info  = KNOWLEDGE_MODES[st.session_state.knowledge_mode]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="status-card status-info">{model_info["icon"]} Modelo: <strong>{model_info["label"]}</strong></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="status-card status-info">{mode_info["icon"]} Modo: <strong>{mode_info["label"].split(" ", 1)[1]}</strong></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="status-card status-ready">🎯 Re-ranking ativo</div>', unsafe_allow_html=True)

    st.markdown("### ❓ Faça uma pergunta")

    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            question = st.text_input(
                "Pergunta",
                placeholder="Ex: Qual é o tema principal do documento?",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("🔍 Buscar", use_container_width=True, type="primary")

    # ── Processar pergunta ──
    if submitted and question.strip():
        # Cache semântico: usa embeddings para encontrar perguntas similares
        cached_result = st.session_state.cache.get_semantic(question) if hasattr(st.session_state.cache, 'get_semantic') else st.session_state.cache.get(question)
        from_cache = cached_result is not None

        if from_cache:
            result        = cached_result
            response_time = 0.0
            model_label   = 'cache'
        else:
            with st.spinner("Buscando e processando sua pergunta..."):
                start  = time.time()
                result = st.session_state.rag_system.query(
                    question,
                    knowledge_mode=st.session_state.knowledge_mode
                )
                response_time = time.time() - start

                st.session_state.cache.set(
                    question=question,
                    answer=result["answer"],
                    sources=result["sources"]
                )
            model_label = st.session_state.selected_model

        interaction_id = st.session_state.db.save_interaction(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            model_used=model_label
        )

        st.session_state.last_result = {
            "question":       question,
            "answer":         result["answer"],
            "sources":        result["sources"],
            "interaction_id": interaction_id,
            "response_time":  response_time,
            "from_cache":     from_cache,
        }
        st.session_state.last_interaction_id = interaction_id
        st.session_state.chat_history.append(st.session_state.last_result)

    # ── Exibir resultado ──
    if st.session_state.last_result:
        res = st.session_state.last_result
        iid = res["interaction_id"]

        st.markdown(f"**🧑 Pergunta:** {res['question']}")

        if res["from_cache"]:
            badge = '<span class="time-badge cache-badge">⚡ Cache — instantâneo</span>'
        else:
            badge = f'<span class="time-badge">⏱️ {res["response_time"]:.2f}s</span>'
        st.markdown(badge, unsafe_allow_html=True)

        st.markdown(f'<div class="answer-card">{res["answer"]}</div>', unsafe_allow_html=True)

        if res["sources"]:
            with st.expander(f"📖 {len(res['sources'])} fonte(s) relevante(s)"):
                for i, source in enumerate(res["sources"], 1):
                    st.markdown(f"**Fonte {i}:** `{source.get('source', 'Desconhecido')}`")
                    content = source.get('content', 'Conteúdo não disponível')
                    st.text(content[:500] + ("..." if len(content) > 500 else ""))
                    if i < len(res["sources"]):
                        st.divider()

        # ── Feedback ──
        if iid not in st.session_state.feedback_ratings:
            st.session_state.feedback_ratings[iid] = None

        already_submitted = iid in st.session_state.feedback_submitted

        if already_submitted:
            st.success("✅ Obrigado! Seu feedback foi registrado.")
        else:
            with st.container():
                st.markdown("---")
                st.markdown("#### ⭐ Enviar Feedback")
                st.caption("Opcional — sua avaliação ajuda a melhorar o sistema.")

                current_rating = st.session_state.feedback_ratings[iid]

                st.write("**Esta resposta foi útil?**")
                c1, c2, c3 = st.columns([2, 2, 2])

                with c1:
                    if st.button(
                        "✅ Sim, útil" if current_rating is True else "👍 Sim, útil",
                        key=f"btn_helpful_{iid}", use_container_width=True
                    ):
                        st.session_state.feedback_ratings[iid] = True
                        st.rerun()
                with c2:
                    if st.button(
                        "❌ Não útil" if current_rating is False else "👎 Não útil",
                        key=f"btn_not_helpful_{iid}", use_container_width=True
                    ):
                        st.session_state.feedback_ratings[iid] = False
                        st.rerun()
                with c3:
                    if st.button(
                        "✖ Remover", key=f"btn_clear_{iid}",
                        use_container_width=True, disabled=(current_rating is None)
                    ):
                        st.session_state.feedback_ratings[iid] = None
                        st.rerun()

                if current_rating is True:
                    st.markdown('<div class="status-card status-ready">✅ Marcado como útil</div>', unsafe_allow_html=True)
                elif current_rating is False:
                    st.markdown('<div class="status-card status-error">👎 Marcado como não útil</div>', unsafe_allow_html=True)
                else:
                    st.caption("Nenhuma avaliação selecionada (opcional)")

                st.markdown("---")
                st.write("**Comentário** (opcional):")
                comment = st.text_area(
                    "Comentário", placeholder="Ex: A resposta foi muito genérica...",
                    label_visibility="collapsed", key=f"comment_{iid}", height=100
                )

                st.markdown("---")
                col_send, col_skip = st.columns([3, 1])
                with col_send:
                    if st.button("📤 Enviar Feedback", key=f"send_{iid}", use_container_width=True, type="primary"):
                        has_rating  = current_rating is not None
                        has_comment = comment.strip() != ''
                        if not has_rating and not has_comment:
                            st.error("❌ Adicione uma avaliação ou comentário.")
                        else:
                            try:
                                st.session_state.db.save_feedback(
                                    interaction_id=iid,
                                    is_helpful=current_rating if has_rating else None,
                                    comment=comment.strip() if has_comment else None
                                )
                                st.session_state.feedback_submitted.add(iid)
                                st.success("✅ Feedback enviado! Obrigado.")
                                st.balloons()
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Erro: {str(e)}")
                with col_skip:
                    if st.button("Pular", key=f"skip_{iid}", use_container_width=True):
                        st.session_state.feedback_submitted.add(iid)
                        st.rerun()

    # ── Histórico ──
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### 📝 Histórico da Sessão")
        for item in reversed(st.session_state.chat_history[-5:]):
            if st.session_state.last_result and item["interaction_id"] == st.session_state.last_result["interaction_id"]:
                continue
            with st.expander(f"🕐 {item['question'][:70]}{'...' if len(item['question']) > 70 else ''}"):
                st.markdown(f"**Pergunta:** {item['question']}")
                st.markdown(f"**Resposta:** {item['answer']}")
                if item['sources']:
                    st.caption(f"📎 {len(item['sources'])} fonte(s)")
                label = "Cache ⚡" if item.get("from_cache") else f"{item.get('response_time', 0):.2f}s"
                st.caption(f"⏱️ {label}")

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAINEL INFERIOR — Configurações e Estatísticas
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

# ── Tabs do painel ──
col_tab1, col_tab2, col_spacer = st.columns([1, 1, 4])

with col_tab1:
    settings_btn = st.button(
        "⚙️ Configurações",
        key="toggle_settings",
        use_container_width=True,
        type="primary" if st.session_state.panel_open == 'settings' else "secondary"
    )
with col_tab2:
    stats_btn = st.button(
        "📊 Estatísticas",
        key="toggle_stats",
        use_container_width=True,
        type="primary" if st.session_state.panel_open == 'stats' else "secondary"
    )

if settings_btn:
    st.session_state.panel_open = None if st.session_state.panel_open == 'settings' else 'settings'
    st.rerun()
if stats_btn:
    st.session_state.panel_open = None if st.session_state.panel_open == 'stats' else 'stats'
    st.rerun()

# ── PAINEL: CONFIGURAÇÕES ─────────────────────────────────────────────────────
if st.session_state.panel_open == 'settings':
    with st.container():
        st.markdown("### ⚙️ Configurações do Sistema")
        st.markdown("---")

        col_left, col_right = st.columns(2)

        # ── Seleção de modelo ──
        with col_left:
            st.markdown("#### 🤖 Modelo de Linguagem")
            st.caption("Instale com `ollama pull <nome>` antes de usar.")

            for model_key, model_data in AVAILABLE_MODELS.items():
                is_selected = st.session_state.selected_model == model_key

                with st.container(border=True):
                    c_info, c_btn = st.columns([3, 1])
                    with c_info:
                        ativo = " — 🟢 **ATIVO**" if is_selected else ""
                        st.markdown(f"{model_data['icon']} **{model_data['label']}**{ativo}")
                        st.caption(f"{model_data['desc']} · Velocidade: **{model_data['speed']}**")
                    with c_btn:
                        if is_selected:
                            st.success("Ativo")
                        else:
                            if st.button("Usar", key=f"select_model_{model_key}", use_container_width=True):
                                st.session_state.selected_model = model_key
                                if st.session_state.rag_system:
                                    with st.spinner(f"Carregando {model_data['label']}..."):
                                        try:
                                            st.session_state.rag_system.change_model(model_key)
                                        except Exception as e:
                                            st.error(f"❌ {e}")
                                st.rerun()

        # ── Modo de conhecimento ──
        with col_right:
            st.markdown("#### 🧠 Modo de Conhecimento")
            st.caption("Define como o sistema usa as informações.")

            for mode_key, mode_data in KNOWLEDGE_MODES.items():
                is_selected = st.session_state.knowledge_mode == mode_key

                with st.container(border=True):
                    c_info, c_btn = st.columns([3, 1])
                    with c_info:
                        ativo = " — 🟢 **ATIVO**" if is_selected else ""
                        st.markdown(f"{mode_data['icon']} **{mode_data['label'].split(' ', 1)[1]}**{ativo}")
                        st.caption(mode_data['desc'])
                    with c_btn:
                        if is_selected:
                            st.success("Ativo")
                        else:
                            if st.button("Usar", key=f"select_mode_{mode_key}", use_container_width=True):
                                st.session_state.knowledge_mode = mode_key
                                if st.session_state.rag_system:
                                    st.session_state.rag_system.change_knowledge_mode(mode_key)
                                st.rerun()

            st.markdown("---")
            st.markdown("#### 🗑️ Cache")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("🗑️ Limpar Cache", use_container_width=True):
                    st.session_state.cache.clear()
                    st.success("✅ Cache limpo!")
                    st.rerun()
            with col_c2:
                cache_stats = st.session_state.cache.get_statistics()
                st.metric("Em cache", cache_stats['cached_responses'])

# ── PAINEL: ESTATÍSTICAS / DASHBOARD ─────────────────────────────────────────
elif st.session_state.panel_open == 'stats':
    try:
        import pandas as pd
        import sqlite3 as _sqlite3

        with st.container():
            st.markdown("### 📊 Dashboard de Estatísticas")
            st.markdown("---")

            stats       = st.session_state.db.get_statistics()
            cache_stats = st.session_state.cache.get_statistics()

            # ── KPIs ──
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("📩 Total Perguntas",   stats['total_interactions'])
            k2.metric("👍 Respostas Úteis",   stats['helpful_count'])
            k3.metric("👎 Respostas Ruins",   stats['not_helpful_count'])
            k4.metric("⭐ Taxa de Utilidade", f"{stats['helpful_rate']:.1f}%")
            k5.metric("⚡ Acertos de Cache",  f"{cache_stats['hit_rate']:.1f}%")

            st.markdown("---")
            col_left, col_right = st.columns(2)

            # ── Gráfico de feedback ao longo do tempo ──
            with col_left:
                st.markdown("#### 📈 Feedbacks ao longo do tempo")
                try:
                    conn = _sqlite3.connect('data/rag_feedback.db')

                    df_fb = pd.read_sql_query("""
                        SELECT
                            DATE(f.timestamp) AS dia,
                            SUM(CASE WHEN f.is_helpful = 1 THEN 1 ELSE 0 END) AS uteis,
                            SUM(CASE WHEN f.is_helpful = 0 THEN 1 ELSE 0 END) AS ruins
                        FROM feedback f
                        WHERE f.is_helpful IS NOT NULL
                        GROUP BY dia
                        ORDER BY dia
                    """, conn)
                    conn.close()

                    if not df_fb.empty:
                        df_fb = df_fb.set_index('dia')
                        st.bar_chart(df_fb, color=["#10b981", "#ef4444"])
                    else:
                        st.info("Nenhum feedback registrado ainda.")
                except Exception as e:
                    st.caption(f"Sem dados de feedback: {e}")

            # ── Perguntas por dia ──
            with col_right:
                st.markdown("#### 📅 Perguntas por dia")
                try:
                    conn = _sqlite3.connect('data/rag_feedback.db')
                    df_q = pd.read_sql_query("""
                        SELECT DATE(timestamp) AS dia, COUNT(*) AS total
                        FROM interactions
                        GROUP BY dia
                        ORDER BY dia
                    """, conn)
                    conn.close()

                    if not df_q.empty:
                        df_q = df_q.set_index('dia')
                        st.area_chart(df_q, color="#3b82f6")
                    else:
                        st.info("Nenhuma interação registrada ainda.")
                except Exception as e:
                    st.caption(f"Sem dados de interação: {e}")

            st.markdown("---")
            col_left2, col_right2 = st.columns(2)

            # ── Modelos usados ──
            with col_left2:
                st.markdown("#### 🤖 Modelos utilizados")
                try:
                    conn = _sqlite3.connect('data/rag_feedback.db')
                    df_m = pd.read_sql_query("""
                        SELECT model_used, COUNT(*) AS total
                        FROM interactions
                        GROUP BY model_used
                        ORDER BY total DESC
                    """, conn)
                    conn.close()

                    if not df_m.empty:
                        st.bar_chart(df_m.set_index('model_used'), color="#8b5cf6")
                    else:
                        st.info("Nenhum dado de modelo disponível.")
                except Exception as e:
                    st.caption(f"Sem dados de modelos: {e}")

            # ── Últimas interações ──
            with col_right2:
                st.markdown("#### 🕐 Últimas interações")
                try:
                    conn = _sqlite3.connect('data/rag_feedback.db')
                    df_last = pd.read_sql_query("""
                        SELECT
                            SUBSTR(question, 1, 50) AS pergunta,
                            SUBSTR(answer, 1, 60)   AS resposta,
                            model_used              AS modelo,
                            SUBSTR(timestamp, 1, 16) AS quando
                        FROM interactions
                        ORDER BY timestamp DESC
                        LIMIT 8
                    """, conn)
                    conn.close()

                    if not df_last.empty:
                        st.dataframe(df_last, use_container_width=True, hide_index=True)
                    else:
                        st.info("Nenhuma interação registrada ainda.")
                except Exception as e:
                    st.caption(f"Sem dados: {e}")

            # ── Aprendizado ──
            st.markdown("---")
            st.markdown("#### 🧠 Estado do Aprendizado")
            try:
                from src.learning.learning_store import LearningStore
                _store   = LearningStore()
                _summary = _store.get_summary()
                _rules   = _store.load_rules()
                _exs     = _store.load_examples(limit=5)

                a1, a2 = st.columns(2)
                a1.metric("📌 Regras ativas",  _summary["rules_count"])
                a2.metric("✍️ Exemplos Q&A",   _summary["examples_count"])

                if _rules:
                    with st.expander("📌 Regras ativas"):
                        for r in _rules:
                            st.caption(f"• {r['rule']}")
                if _exs:
                    with st.expander("✍️ Exemplos recentes (few-shot)"):
                        for ex in _exs:
                            st.markdown(f"**Q:** {ex['question'][:80]}")
                            st.caption(f"→ {ex['good_answer'][:120]}...")
                            st.divider()
            except Exception:
                st.caption("Aprendizado não disponível.")

    except ImportError:
        st.error("❌ Instale `pandas` para ver o dashboard: `pip install pandas`")