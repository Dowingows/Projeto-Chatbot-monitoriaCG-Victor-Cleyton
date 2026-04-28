const API = '/api';
let isLoading = false;

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  setupInput();
  setupSuggestions();
  document.getElementById('new-chat-btn').addEventListener('click', newChat);
});

// ── History ───────────────────────────────────────────────────────────────

async function loadHistory() {
  try {
    const res = await fetch(`${API}/history`);
    const { messages } = await res.json();
    if (messages.length > 0) {
      hideWelcome();
      messages.forEach(m => appendMessage(m.role, m.content));
    }
  } catch (_) {}
}

async function newChat() {
  try {
    await fetch(`${API}/history`, { method: 'DELETE' });
  } catch (_) {}
  document.getElementById('messages').innerHTML = '';
  document.getElementById('messages').appendChild(buildWelcome());
  setupSuggestions();
}

// ── Send ──────────────────────────────────────────────────────────────────

async function sendMessage() {
  if (isLoading) return;
  const input = document.getElementById('message-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  autoResize(input);
  updateSendBtn();
  hideWelcome();

  appendMessage('user', text);
  const loadingEl = appendLoading();

  isLoading = true;
  updateSendBtn();

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });

    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    loadingEl.remove();
    appendMessage('assistant', data.response, {
      topic: data.topic,
      search_type: data.search_type,
      sources: data.sources,
      out_of_scope: data.out_of_scope,
    });
  } catch (_) {
    loadingEl.remove();
    appendMessage('assistant', 'Erro ao conectar com o servidor. Verifique se o Ollama está em execução.');
  } finally {
    isLoading = false;
    updateSendBtn();
  }
}

// ── DOM builders ──────────────────────────────────────────────────────────

function appendMessage(role, content, meta = {}) {
  const messages = document.getElementById('messages');
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  wrap.appendChild(bubble);

  if (role === 'assistant' && !meta.out_of_scope && meta.topic) {
    const metaEl = document.createElement('div');
    metaEl.className = 'message-meta';

    const topicBadge = document.createElement('span');
    topicBadge.className = 'badge badge-topic';
    topicBadge.textContent = meta.topic;
    metaEl.appendChild(topicBadge);

    if (meta.search_type) {
      const searchBadge = document.createElement('span');
      searchBadge.className = 'badge badge-search';
      searchBadge.textContent = meta.search_type === 'filtrado' ? 'busca filtrada' : 'busca global';
      metaEl.appendChild(searchBadge);
    }

    wrap.appendChild(metaEl);

    if (meta.sources && meta.sources.length > 0) {
      wrap.appendChild(buildSources(meta.sources));
    }
  }

  messages.appendChild(wrap);
  scrollToBottom();
}

function appendLoading() {
  const messages = document.getElementById('messages');
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
  messages.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function buildSources(sources) {
  const container = document.createElement('div');

  const toggle = document.createElement('button');
  toggle.className = 'sources-toggle';
  toggle.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
    ${sources.length} fonte${sources.length > 1 ? 's' : ''}
  `;

  const list = document.createElement('div');
  list.className = 'sources-list';
  sources.forEach(s => {
    const item = document.createElement('div');
    item.className = 'source-item';
    item.textContent = s;
    list.appendChild(item);
  });

  toggle.addEventListener('click', () => {
    toggle.classList.toggle('open');
    list.classList.toggle('open');
  });

  container.appendChild(toggle);
  container.appendChild(list);
  return container;
}

function buildWelcome() {
  const el = document.createElement('div');
  el.className = 'welcome';
  el.id = 'welcome';
  el.innerHTML = `
    <div class="welcome-logo">Luan<span>.AI</span></div>
    <p>Olá! Sou a assistente da disciplina de <strong>Computação Gráfica</strong>.<br>Como posso ajudar você hoje?</p>
    <div class="suggestions">
      <button class="suggestion-btn">O que é pipeline gráfico?</button>
      <button class="suggestion-btn">Como funcionam as transformações 2D?</button>
      <button class="suggestion-btn">Explique o modelo de iluminação de Phong</button>
      <button class="suggestion-btn">O que é Viewing 3D?</button>
    </div>
  `;
  return el;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function hideWelcome() {
  document.getElementById('welcome')?.remove();
}

function scrollToBottom() {
  const messages = document.getElementById('messages');
  messages.scrollTop = messages.scrollHeight;
}

function updateSendBtn() {
  const input = document.getElementById('message-input');
  const btn = document.getElementById('send-btn');
  btn.disabled = isLoading || input.value.trim() === '';
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function setupInput() {
  const input = document.getElementById('message-input');
  input.addEventListener('input', () => { autoResize(input); updateSendBtn(); });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  document.getElementById('send-btn').addEventListener('click', sendMessage);
}

function setupSuggestions() {
  document.querySelectorAll('.suggestion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.getElementById('message-input');
      input.value = btn.textContent;
      autoResize(input);
      updateSendBtn();
      sendMessage();
    });
  });
}
