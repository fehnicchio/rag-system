// static/js/app.js
// Frontend do RAG Q&A System — consome a API REST exposta por server.py

const API = "";

const el = (id) => document.getElementById(id);

const state = {
  models: {},
  modes: {},
  selectedModel: "mistral",
  selectedMode: "docs_only",
  systemReady: false,
  feedbackState: {}, // interaction_id -> {rating: true|false|null, submitted: bool}
};

// ─── Helpers ────────────────────────────────────────────────────────────────

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* sem corpo */ }
  if (!res.ok) {
    const message = (data && data.detail) ? data.detail : `Erro ${res.status}`;
    throw new Error(message);
  }
  return data;
}

function toast(message, isError = false) {
  const t = el("toast");
  t.textContent = message;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => t.classList.remove("show"), 3200);
}

function renderMarkdown(text) {
  try {
    const raw = marked.parse(text || "");
    return DOMPurify.sanitize(raw);
  } catch (_) {
    return (text || "").replace(/</g, "&lt;");
  }
}

function escapeHtml(str) {
  return (str || "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// ─── Status ────────────────────────────────────────────────────────────────

async function refreshStatus() {
  const status = await api("/api/status");
  state.systemReady = status.system_ready;

  const dot = el("statusBlock").querySelector(".status-dot");
  const text = el("statusText");
  dot.className = "status-dot";

  if (status.system_ready) {
    dot.classList.add("status-dot--ready");
    text.textContent = "Sistema pronto e operacional";
  } else if (status.initialized) {
    dot.classList.add("status-dot--warn");
    text.textContent = "Iniciado — construa a base de conhecimento";
  } else {
    dot.classList.add("status-dot--off");
    text.textContent = "Sistema não inicializado";
  }

  el("initBtn").textContent = status.initialized ? "Reinicializar sistema" : "Inicializar sistema";
  el("docsCount").textContent = status.documents_count;

  const docList = el("docList");
  if (status.documents.length) {
    docList.innerHTML = status.documents.map((d) => `<div class="doc-item" title="${escapeHtml(d)}">${escapeHtml(d)}</div>`).join("");
  } else {
    docList.innerHTML = `<p class="muted small">Nenhum documento ainda.</p>`;
  }

  state.selectedModel = status.model;
  state.selectedMode = status.knowledge_mode;
  updateBadges();

  el("sendBtn").disabled = false; // permitido tentar; backend valida
  el("composerForm").querySelector("textarea").placeholder = status.system_ready
    ? "Faça uma pergunta sobre seus documentos…"
    : "Inicialize o sistema e construa a base para começar…";
}

function updateBadges() {
  const model = state.models[state.selectedModel];
  const mode = state.modes[state.selectedMode];
  if (model) el("modelBadge").textContent = `${model.icon} ${model.label}`;
  if (mode) el("modeBadge").textContent = `${mode.icon} ${mode.label}`;
}

async function loadModelsAndModes() {
  const m = await api("/api/models");
  state.models = m.models;
  state.selectedModel = m.selected;

  const k = await api("/api/knowledge-modes");
  state.modes = k.modes;
  state.selectedMode = k.selected;

  await renderModelList();
  renderModeList();
  updateBadges();
}

// ─── Inicializar / construir base ──────────────────────────────────────────

el("initBtn").addEventListener("click", async () => {
  const btn = el("initBtn");
  btn.disabled = true;
  btn.textContent = "Inicializando…";
  try {
    await api("/api/initialize", { method: "POST", body: JSON.stringify({ model_name: state.selectedModel }) });
    toast("Sistema inicializado!");
    await refreshStatus();
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
});

el("buildBtn").addEventListener("click", async () => {
  const btn = el("buildBtn");
  btn.disabled = true;
  btn.textContent = "Construindo…";
  try {
    await api("/api/build", { method: "POST" });
    toast("Base de conhecimento construída!");
    await refreshStatus();
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Construir base";
  }
});

// ─── Upload de documentos ───────────────────────────────────────────────────

el("fileInput").addEventListener("change", async (ev) => {
  const files = ev.target.files;
  if (!files || !files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  try {
    const result = await api("/api/documents/upload", { method: "POST", body: fd });
    toast(`${result.count} arquivo(s) salvo(s). Construa a base para incluí-los.`);
    await refreshStatus();
  } catch (e) {
    toast(e.message, true);
  } finally {
    ev.target.value = "";
  }
});

// ─── Scraping de URL ────────────────────────────────────────────────────────

el("scrapeBtn").addEventListener("click", async () => {
  const input = el("scrapeUrl");
  const url = input.value.trim();
  if (!url) { toast("Digite uma URL válida", true); return; }
  const btn = el("scrapeBtn");
  btn.disabled = true;
  try {
    const result = await api("/api/scrape", { method: "POST", body: JSON.stringify({ url }) });
    toast(`Importado: ${result.file}. Construa a base para incluí-lo.`);
    input.value = "";
    await refreshStatus();
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
});

// ─── Aprendizado contínuo ───────────────────────────────────────────────────

async function refreshLearning() {
  try {
    const summary = await api("/api/learning/summary");
    if (!summary.available) return;
    el("rulesCount").textContent = summary.rules_count;
    el("examplesCount").textContent = summary.examples_count;
    const pendingRow = el("pendingRow");
    if (summary.pending_feedback > 0) {
      pendingRow.style.display = "flex";
      el("pendingCount").textContent = summary.pending_feedback;
    } else {
      pendingRow.style.display = "none";
    }
  } catch (_) { /* aprendizado ainda não inicializado */ }
}

el("learnBtn").addEventListener("click", async () => {
  const btn = el("learnBtn");
  btn.disabled = true;
  btn.textContent = "Analisando…";
  try {
    const result = await api("/api/learning/analyze", { method: "POST" });
    if (result.processed === 0) {
      toast("Nenhum feedback novo para analisar.");
    } else {
      toast(`${result.processed} feedback(s) processado(s) · ${result.new_rules.length} regra(s) nova(s)`);
    }
    await refreshLearning();
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analisar e aprender";
  }
});

// ─── Limpar sessão ──────────────────────────────────────────────────────────

el("clearBtn").addEventListener("click", () => {
  el("messages").innerHTML = "";
  el("emptyState").style.display = "flex";
  toast("Histórico da sessão limpo.");
});

// ─── Chat ───────────────────────────────────────────────────────────────────

const textarea = el("questionInput");
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px";
});
textarea.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    el("composerForm").requestSubmit();
  }
});

el("composerForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const question = textarea.value.trim();
  if (!question) return;

  el("emptyState").style.display = "none";
  appendUserMessage(question);
  textarea.value = "";
  textarea.style.height = "auto";

  const sendBtn = el("sendBtn");
  sendBtn.disabled = true;
  const placeholderId = appendPendingAssistant();

  try {
    const result = await api("/api/query", {
      method: "POST",
      body: JSON.stringify({ question, knowledge_mode: state.selectedMode }),
    });
    replacePendingAssistant(placeholderId, result);
    refreshLearning();
  } catch (e) {
    replacePendingAssistant(placeholderId, { answer: `⚠️ ${e.message}`, sources: [], from_cache: false, response_time: 0, interaction_id: null, error: true });
  } finally {
    sendBtn.disabled = false;
  }
});

function appendUserMessage(question) {
  const wrap = document.createElement("div");
  wrap.className = "msg-user";
  wrap.textContent = question;
  el("messages").appendChild(wrap);
  scrollToBottom();
}

let pendingCounter = 0;
function appendPendingAssistant() {
  const id = `pending-${++pendingCounter}`;
  const wrap = document.createElement("div");
  wrap.className = "msg-assistant";
  wrap.id = id;
  wrap.innerHTML = `
    <div class="msg-avatar">R</div>
    <div class="msg-body">
      <div class="answer-card"><em class="muted">Buscando e processando sua pergunta…</em></div>
    </div>`;
  el("messages").appendChild(wrap);
  scrollToBottom();
  return id;
}

function replacePendingAssistant(id, result) {
  const wrap = el(id);
  if (!wrap) return;

  const badge = result.error
    ? ""
    : result.from_cache
      ? `<span class="time-badge time-badge--cache">⚡ Cache${result.similarity ? ` · ${(result.similarity * 100).toFixed(0)}% similar` : ""}</span>`
      : `<span class="time-badge">⏱ ${result.response_time.toFixed(2)}s</span>`;

  const sourcesHtml = (result.sources && result.sources.length)
    ? `<details class="sources"><summary>${result.sources.length} fonte(s) relevante(s)</summary>
        ${result.sources.map((s) => `
          <div class="source-item">
            <div class="source-name">${escapeHtml(s.source || "Desconhecido")}</div>
            <div class="source-text">${escapeHtml((s.content || "").slice(0, 500))}${(s.content || "").length > 500 ? "…" : ""}</div>
          </div>`).join("")}
      </details>`
    : "";

  const feedbackHtml = (result.interaction_id && !result.error)
    ? `<div class="feedback" data-iid="${result.interaction_id}">
        <div class="feedback-row">
          <span class="label">Esta resposta foi útil?</span>
          <button class="fb-btn" data-fb="yes">👍 Sim</button>
          <button class="fb-btn" data-fb="no">👎 Não</button>
        </div>
        <textarea placeholder="Comentário opcional…"></textarea>
        <div class="feedback-row" style="margin-top:8px;">
          <button class="btn btn-outline" data-fb="send">Enviar feedback</button>
        </div>
      </div>`
    : "";

  wrap.querySelector(".msg-body").innerHTML = `
    <div class="msg-meta">${badge}</div>
    <div class="answer-card">${renderMarkdown(result.answer)}</div>
    ${sourcesHtml}
    ${feedbackHtml}
  `;

  if (result.interaction_id) wireFeedback(wrap.querySelector(".feedback"));
  scrollToBottom();
}

function wireFeedback(container) {
  if (!container) return;
  const iid = container.dataset.iid;
  let rating = null;

  container.querySelectorAll("[data-fb]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const action = btn.dataset.fb;

      if (action === "yes" || action === "no") {
        rating = action === "yes";
        container.querySelectorAll('[data-fb="yes"],[data-fb="no"]').forEach((b) => {
          b.classList.remove("active-yes", "active-no");
        });
        btn.classList.add(action === "yes" ? "active-yes" : "active-no");
        return;
      }

      if (action === "send") {
        const comment = container.querySelector("textarea").value.trim();
        if (rating === null && !comment) {
          toast("Adicione uma avaliação ou comentário.", true);
          return;
        }
        try {
          await api("/api/feedback", {
            method: "POST",
            body: JSON.stringify({ interaction_id: Number(iid), is_helpful: rating, comment }),
          });
          container.innerHTML = `<span class="feedback-submitted">✅ Obrigado! Seu feedback foi registrado.</span>`;
          refreshLearning();
        } catch (e) {
          toast(e.message, true);
        }
      }
    });
  });
}

function scrollToBottom() {
  const box = el("chatScroll");
  box.scrollTop = box.scrollHeight;
}

// ─── Modal: Configurações ───────────────────────────────────────────────────

async function renderModelList() {
  const list = el("modelList");
  list.innerHTML = `<p class="muted small">Verificando modelos instalados…</p>`;

  let modelsData = state.models;

  // Buscar status de instalação
  try {
    const checked = await api("/api/models/check");
    modelsData = checked.models;
    state.models = checked.models;
  } catch (_) {
    // Se falhar, usa os dados já carregados sem status de instalação
  }

  list.innerHTML = Object.entries(modelsData).map(([key, m]) => {
    const isSelected  = key === state.selectedModel;
    const isInstalled = m.installed !== false; // true se não tiver o campo

    return `
      <div class="option-card ${isSelected ? "selected" : ""} ${!isInstalled ? "option-card--dimmed" : ""}"
          data-model="${key}" data-installed="${isInstalled}">
        <div class="option-title">
          ${m.icon} ${m.label}
          ${isSelected   ? '<span class="active-pill">ATIVO</span>' : ""}
          ${isInstalled  ? '<span class="pill-installed">✅ Instalado</span>' : '<span class="pill-missing">⚠️ Não instalado</span>'}
        </div>
        <div class="option-desc">${m.desc} · <strong>Velocidade:</strong> ${m.speed}</div>
        ${!isInstalled ? `<div class="install-hint">💡 <code>ollama pull ${key}</code></div>` : ""}
      </div>`;
  }).join("");

  list.querySelectorAll("[data-model]").forEach((card) => {
    card.addEventListener("click", async () => {
      const model     = card.dataset.model;
      const installed = card.dataset.installed === "true";

      if (model === state.selectedModel) return;

      if (!installed) {
        toast(`Instale o modelo primeiro: ollama pull ${model}`, true);
        return;
      }

      try {
        await api("/api/settings/model", { method: "POST", body: JSON.stringify({ model }) });
        state.selectedModel = model;
        await renderModelList();
        updateBadges();
        toast(`Modelo alterado para ${state.models[model].label}`);
      } catch (e) {
        toast(e.message, true);
      }
    });
  });
}

async function refreshCacheStats() {
  try {
    const stats = await api("/api/cache/stats");
    el("cacheCount").textContent = stats.cached_responses;
    el("cacheHitRate").textContent = `${stats.hit_rate.toFixed(1)}%`;
  } catch (_) { /* ignore */ }
}

el("clearCacheBtn").addEventListener("click", async () => {
  try {
    await api("/api/cache/clear", { method: "POST" });
    toast("Cache limpo!");
    refreshCacheStats();
  } catch (e) {
    toast(e.message, true);
  }
});

// ─── Modal: Estatísticas ────────────────────────────────────────────────────

function renderBarChart(container, data, labelKey, valueKeys, colors) {
  if (!data.length) {
    container.innerHTML = `<p class="muted small">Sem dados ainda.</p>`;
    return;
  }
  const max = Math.max(1, ...data.flatMap((d) => valueKeys.map((k) => Number(d[k]) || 0)));
  container.innerHTML = data.map((d) => {
    const bars = valueKeys.map((k, i) => {
      const val = Number(d[k]) || 0;
      const heightPct = Math.max(2, (val / max) * 100);
      return `<div class="bar" title="${k}: ${val}" style="height:${heightPct}%; background:${colors[i]};"></div>`;
    }).join("");
    return `<div class="bar-col">${bars}<span class="bar-label">${escapeHtml(String(d[labelKey]).slice(5))}</span></div>`;
  }).join("");
}

async function openStatsModal() {
  el("statsModal").classList.add("open");
  try {
    const dash = await api("/api/dashboard");
    el("kpiTotal").textContent = dash.kpis.total_interactions;
    el("kpiHelpful").textContent = dash.kpis.helpful_count;
    el("kpiNotHelpful").textContent = dash.kpis.not_helpful_count;
    el("kpiHelpfulRate").textContent = `${dash.kpis.helpful_rate.toFixed(1)}%`;
    el("kpiCacheRate").textContent = `${dash.cache.hit_rate.toFixed(1)}%`;

    renderBarChart(el("chartQuestions"), dash.questions_per_day, "day", ["total"], ["#CC785C"]);
    renderBarChart(el("chartModels"), dash.models_used, "model_used", ["total"], ["#CC785C"]);

    const tbody = el("recentTable").querySelector("tbody");
    tbody.innerHTML = dash.recent_interactions.map((r) => `
      <tr>
        <td>${escapeHtml(r.question)}</td>
        <td>${escapeHtml(r.answer)}</td>
        <td>${escapeHtml(r.model_used || "")}</td>
        <td>${escapeHtml((r.timestamp || "").slice(0, 16))}</td>
      </tr>`).join("") || `<tr><td colspan="4" class="muted">Nenhuma interação registrada ainda.</td></tr>`;
  } catch (e) {
    toast(e.message, true);
  }
}

// ─── Modais: abrir/fechar ───────────────────────────────────────────────────

el("settingsBtn").addEventListener("click", () => {
  el("settingsModal").classList.add("open");
  refreshCacheStats();
});
el("statsBtn").addEventListener("click", openStatsModal);

document.querySelectorAll("[data-close-modal]").forEach((btn) => {
  btn.addEventListener("click", (ev) => {
    ev.target.closest(".modal-backdrop").classList.remove("open");
  });
});
document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
  backdrop.addEventListener("click", (ev) => {
    if (ev.target === backdrop) backdrop.classList.remove("open");
  });
});

// ─── Sidebar mobile toggle ──────────────────────────────────────────────────

el("sidebarToggle").addEventListener("click", () => {
  el("sidebar").classList.toggle("open");
});

// ─── Boot ───────────────────────────────────────────────────────────────────

(async function init() {
  try {
    await loadModelsAndModes();
    await refreshStatus();
    await refreshLearning();
  } catch (e) {
    toast(`Erro ao carregar: ${e.message}`, true);
  }
})();
