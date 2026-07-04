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
    /* Importar fonte Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Cabeçalho principal ── */
    .rag-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
        color: white;
    }
    .rag-header h1 {
        margin: 0 0 4px 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .rag-header p {
        margin: 0;
        opacity: 0.8;
        font-size: 0.95rem;
    }

    /* ── Cards de status ── */
    .status-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-ready {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .status-error {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .status-warn {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fcd34d;
    }

    /* ── Card de resposta ── */
    .answer-card {
        background: #f8faff;
        border: 1px solid #dbeafe;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 20px 24px;
        line-height: 1.7;
        font-size: 0.97rem;
        color: #1e293b;
        margin: 12px 0;
    }

    /* ── Seção de feedback ── */
    .feedback-section {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .feedback-section h4 {
        margin: 0 0 16px 0;
        font-size: 0.95rem;
        color: #374151;
        font-weight: 600;
    }

    /* ── Rating ativo ── */
    .rating-selected-good {
        background: #ecfdf5 !important;
        border: 2px solid #10b981 !important;
        color: #065f46 !important;
        font-weight: 600;
    }
    .rating-selected-bad {
        background: #fef2f2 !important;
        border: 2px solid #ef4444 !important;
        color: #991b1b !important;
        font-weight: 600;
    }

    /* ── Badge de tempo ── */
    .time-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #64748b;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 6px;
        font-weight: 500;
    }
    .cache-badge {
        background: #fef9c3;
        color: #854d0e;
    }

    /* ── Métricas da sidebar ── */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
    }

    /* ── Histórico ── */
    .history-item {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .history-q {
        font-weight: 600;
        color: #1e3a5f;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .history-a {
        color: #374151;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Botão primário customizado ── */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        opacity: 0.9;
    }

    /* Esconder label do text_input quando collapsed */
    .stTextInput > label[data-testid="stWidgetLabel"] {
        display: none;
    }
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
    st.session_state.initialized = True
    st.session_state.rag_system = None
    st.session_state.chat_history = []
    st.session_state.db = RAGDatabase()
    st.session_state.cache = CacheManager()
    st.session_state.last_result = None          # Guarda último resultado
    st.session_state.last_interaction_id = None
    st.session_state.feedback_submitted = set()  # IDs que já tiveram feedback enviado
    st.session_state.feedback_ratings = {}       # {interaction_id: True/False/None}
    st.session_state.url_input_key = 0

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuração")

    # Status
    status = st.session_state.rag_system.get_status() if st.session_state.rag_system else None

    if status and status['system_ready']:
        st.markdown('<div class="status-card status-ready">✅ Sistema pronto e operacional</div>', unsafe_allow_html=True)
    elif status:
        st.markdown('<div class="status-card status-warn">⚠️ Sistema iniciado, mas índice não construído</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card status-error">❌ Sistema não inicializado</div>', unsafe_allow_html=True)

    st.divider()

    # Inicializar sistema
    if st.button("🚀 Inicializar Sistema", use_container_width=True, type="primary"):
        try:
            with st.spinner("Carregando modelos e índices..."):
                st.session_state.rag_system = RAGSystem()
            st.success("✅ Sistema inicializado com sucesso!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erro ao inicializar: {str(e)}")

    st.divider()

    # ── Base de conhecimento ──
    st.markdown("#### 📄 Base de Conhecimento")

    doc_path = 'data/documents'
    docs = list_documents(doc_path)

    if docs:
        st.info(f"📦 {len(docs)} documento(s) encontrado(s)")
        with st.expander("📋 Ver lista de documentos"):
            for doc in docs:
                st.caption(f"• {os.path.basename(doc)}")
    else:
        st.warning("Nenhum documento em `data/documents/`")

    # Abas de adição de documentos
    tab1, tab2 = st.tabs(["📁 Arquivos Locais", "🌐 Web"])

    with tab1:
        if st.button("🔨 Construir Base de Conhecimento", use_container_width=True):
            if not st.session_state.rag_system:
                st.error("Inicialize o sistema primeiro!")
            else:
                try:
                    with st.spinner("Processando documentos e construindo índice..."):
                        success = st.session_state.rag_system.build_knowledge_base()
                    if success:
                        st.success("✅ Base construída com sucesso!")
                        st.rerun()
                    else:
                        st.error("❌ Falha ao construir a base. Verifique os documentos.")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

        # Upload de arquivos
        st.markdown("**Upload de arquivos:**")
        uploaded_files = st.file_uploader(
            "PDF, TXT, DOCX, MD",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            if st.button(f"💾 Salvar {len(uploaded_files)} arquivo(s)", use_container_width=True):
                try:
                    for uploaded_file in uploaded_files:
                        dest = Path('data/documents') / uploaded_file.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with open(dest, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                    st.success(f"✅ {len(uploaded_files)} arquivo(s) salvo(s)! Reconstrua a base para aplicar.")
                except Exception as e:
                    st.error(f"❌ Erro ao salvar: {str(e)}")

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
                    with st.spinner("Extraindo conteúdo da URL..."):
                        scraper = WebScraper()
                        scraped_data = scraper.scrape_url(url_input)

                    if scraped_data:
                        filepath = scraper.save_to_document(scraped_data)
                        st.session_state.url_input_key += 1
                        st.success(f"✅ Salvo: `{os.path.basename(filepath)}`")
                        st.info("Reconstrua a base para incluir este documento.")
                    else:
                        st.error("❌ Falha ao extrair conteúdo. Verifique a URL.")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

        st.caption("Cole qualquer URL (artigos, blogs, documentação). O texto será extraído automaticamente.")

    st.divider()

    # ── Estatísticas ──
    st.markdown("#### 📊 Estatísticas")
    stats = st.session_state.db.get_statistics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Perguntas", stats['total_interactions'])
    with col2:
        st.metric("Úteis", f"{stats['helpful_rate']:.0f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("👍 Úteis", stats['helpful_count'])
    with col2:
        st.metric("👎 Ruins", stats['not_helpful_count'])

    st.divider()

    # ── Cache ──
    st.markdown("#### ⚡ Cache")
    cache_stats = st.session_state.cache.get_statistics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Em cache", cache_stats['cached_responses'])
    with col2:
        st.metric("Acertos", f"{cache_stats['hit_rate']:.0f}%")

    if st.button("🗑️ Limpar Cache", use_container_width=True):
        st.session_state.cache.clear()
        st.success("✅ Cache limpo!")
        st.rerun()

    st.divider()

    # ── Re-ranking ──
    st.markdown("#### 🎯 Re-ranking")
    st.markdown('<div class="status-card status-ready">✅ Ativo — busca top-10, retorna top-3</div>', unsafe_allow_html=True)

    st.divider()

    # ── Aprendizado ──
    st.markdown("#### 🧠 Aprendizado")
    if st.button("📊 Analisar Feedback", use_container_width=True):
        try:
            from src.learning.feedback_learner import FeedbackLearner
            with st.spinner("Analisando feedbacks negativos..."):
                learner = FeedbackLearner()
                analysis = learner.analyze_feedback()
                learner.save_recommendations(analysis['recommendations'])

            if analysis['total_analyzed'] > 0:
                st.metric("Feedbacks analisados", analysis['total_analyzed'])
                if analysis['issues_found']:
                    st.write("**🔍 Principais problemas:**")
                    for issue, data in analysis['issues_found'][:3]:
                        with st.expander(f"{issue} ({data['count']}x)"):
                            for ex in data['examples'][:2]:
                                st.caption(f"**Q:** {ex['question']}")
                                st.caption(f"**Feedback:** {ex['comment']}")
            else:
                st.info("Nenhum feedback negativo ainda.")
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")

    st.divider()

    if st.button("🗑️ Limpar Histórico da Sessão", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.session_state.last_interaction_id = None
        st.success("Histórico da sessão limpo!")

# ─── Área principal ───────────────────────────────────────────────────────────

if not st.session_state.rag_system or not st.session_state.rag_system.get_status()['system_ready']:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
        <div style="font-size: 3rem; margin-bottom: 12px;">👈</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #475569;">Complete a configuração na barra lateral</div>
        <div style="font-size: 0.9rem; margin-top: 8px;">Inicialize o sistema e construa a base de conhecimento para começar</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Campo de pergunta ──
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
        cached_result = st.session_state.cache.get(question)
        from_cache = cached_result is not None

        if from_cache:
            result = cached_result
            response_time = 0.0
            model_label = 'cache'
        else:
            with st.spinner("Buscando e processando sua pergunta..."):
                start = time.time()
                result = st.session_state.rag_system.query(question)
                response_time = time.time() - start

                st.session_state.cache.set(
                    question=question,
                    answer=result["answer"],
                    sources=result["sources"]
                )
            model_label = 'mistral'

        # Salvar interação no banco
        interaction_id = st.session_state.db.save_interaction(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            model_used=model_label
        )

        # Atualizar session state com o resultado atual
        st.session_state.last_result = {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "interaction_id": interaction_id,
            "response_time": response_time,
            "from_cache": from_cache,
        }
        st.session_state.last_interaction_id = interaction_id

        # Adicionar ao histórico
        st.session_state.chat_history.append(st.session_state.last_result)

    # ─────────────────────────────────────────────────────────────────────────
    # EXIBIR RESULTADO — fora do bloco `if submitted`, para sobreviver ao rerun
    # causado pelos botões de feedback
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.last_result:
        res = st.session_state.last_result
        iid = res["interaction_id"]

        # Pergunta feita
        st.markdown(f"**🧑 Pergunta:** {res['question']}")

        # Badge de tempo
        if res["from_cache"]:
            badge = f'<span class="time-badge cache-badge">⚡ Cache — instantâneo</span>'
        else:
            badge = f'<span class="time-badge">⏱️ {res["response_time"]:.2f}s</span>'
        st.markdown(badge, unsafe_allow_html=True)

        # Resposta
        st.markdown(f'<div class="answer-card">{res["answer"]}</div>', unsafe_allow_html=True)

        # ── Fontes ──
        if res["sources"]:
            with st.expander(f"📖 {len(res['sources'])} fonte(s) relevante(s)"):
                for i, source in enumerate(res["sources"], 1):
                    source_name = source.get('source', 'Desconhecido')
                    source_content = source.get('content', 'Conteúdo não disponível')
                    st.markdown(f"**Fonte {i}:** `{source_name}`")
                    st.text(source_content[:500] + ("..." if len(source_content) > 500 else ""))
                    if i < len(res["sources"]):
                        st.divider()

        # ── Feedback ──────────────────────────────────────────────────────────
        # Inicializar rating no session state (persiste entre reruns)
        if iid not in st.session_state.feedback_ratings:
            st.session_state.feedback_ratings[iid] = None

        already_submitted = iid in st.session_state.feedback_submitted

        if already_submitted:
            st.success("✅ Obrigado! Seu feedback foi registrado.")
        else:
            st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
            st.markdown("#### ⭐ Enviar Feedback")
            st.caption("Opcional — sua avaliação ajuda a melhorar o sistema.")

            # ── Passo 1: Avaliação (botões NÃO submetem feedback, só selecionam) ──
            current_rating = st.session_state.feedback_ratings[iid]

            st.write("**Esta resposta foi útil?**")
            c1, c2, c3 = st.columns([2, 2, 2])

            with c1:
                # Botão Sim — apenas atualiza o rating no state e reroda
                btn_helpful = st.button(
                    "👍  Sim, útil" if current_rating is not True else "✅  Sim, útil",
                    key=f"btn_helpful_{iid}",
                    use_container_width=True
                )
            with c2:
                btn_not_helpful = st.button(
                    "👎  Não útil" if current_rating is not False else "❌  Não útil",
                    key=f"btn_not_helpful_{iid}",
                    use_container_width=True
                )
            with c3:
                btn_clear = st.button(
                    "✖ Remover",
                    key=f"btn_clear_{iid}",
                    use_container_width=True,
                    disabled=(current_rating is None)
                )

            # Processar cliques — apenas atualiza state, NÃO envia feedback
            if btn_helpful:
                st.session_state.feedback_ratings[iid] = True
                st.rerun()
            elif btn_not_helpful:
                st.session_state.feedback_ratings[iid] = False
                st.rerun()
            elif btn_clear:
                st.session_state.feedback_ratings[iid] = None
                st.rerun()

            # Indicador visual da seleção atual
            current_rating = st.session_state.feedback_ratings[iid]  # Reler após rerun
            if current_rating is True:
                st.markdown('<div class="status-card status-ready">✅ Marcado como útil</div>', unsafe_allow_html=True)
            elif current_rating is False:
                st.markdown('<div class="status-card status-error">👎 Marcado como não útil</div>', unsafe_allow_html=True)
            else:
                st.caption("Nenhuma avaliação selecionada (opcional)")

            st.markdown("---")

            # ── Passo 2: Comentário ──
            st.write("**Comentário** (opcional):")
            comment = st.text_area(
                "Comentário",
                placeholder="Ex: A resposta foi muito genérica, gostaria de mais exemplos práticos...",
                label_visibility="collapsed",
                key=f"comment_{iid}",
                height=100
            )

            st.markdown("---")

            # ── Passo 3: Botão de envio ──
            col_send, col_skip = st.columns([3, 1])
            with col_send:
                send_btn = st.button(
                    "📤 Enviar Feedback",
                    key=f"send_feedback_{iid}",
                    use_container_width=True,
                    type="primary"
                )
            with col_skip:
                skip_btn = st.button(
                    "Pular",
                    key=f"skip_feedback_{iid}",
                    use_container_width=True
                )

            if send_btn:
                has_rating = current_rating is not None
                has_comment = comment.strip() != ''

                if not has_rating and not has_comment:
                    st.error("❌ Adicione uma avaliação ou comentário antes de enviar.")
                else:
                    try:
                        # FIX: Salvar feedback em UMA única chamada consolidada
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
                        st.error(f"❌ Erro ao enviar feedback: {str(e)}")

            if skip_btn:
                st.session_state.feedback_submitted.add(iid)
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

# ─── Histórico da sessão ──────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.divider()
    st.markdown("### 📝 Histórico da Sessão")

    # Mostrar últimas 5, mais recente primeiro
    history_to_show = list(reversed(st.session_state.chat_history[-5:]))

    for i, item in enumerate(history_to_show):
        # Não repetir o item atual (já exibido acima)
        if (st.session_state.last_result
                and item["interaction_id"] == st.session_state.last_result["interaction_id"]
                and i == 0):
            continue

        with st.expander(f"🕐 {item['question'][:70]}{'...' if len(item['question']) > 70 else ''}"):
            st.markdown(f"**Pergunta:** {item['question']}")
            st.markdown(f"**Resposta:** {item['answer']}")
            if item['sources']:
                st.caption(f"📎 {len(item['sources'])} fonte(s) consultada(s)")
            time_label = "Cache" if item.get("from_cache") else f"{item.get('response_time', 0):.2f}s"
            st.caption(f"⏱️ {time_label}")