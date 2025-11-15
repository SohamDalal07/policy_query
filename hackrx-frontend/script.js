const API_BASE = "http://127.0.0.1:8000";
let serverOnline = false;
let sessionToken = "";
const THEME_KEY = "HACKRX_THEME";

function getToken() {
  return sessionToken;
}

function saveToken() {
  const tokenField = document.getElementById("tokenInput");
  const token = tokenField.value.trim();
  if (!token) {
    alert("Please enter a token");
    return;
  }
  sessionToken = token;
  tokenField.value = "";
  alert("Token stored for this session (clears on refresh)");
}

function addMessage(text, sender) {
  const chatWindow = document.getElementById("chatWindow");
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender);
  messageDiv.textContent = text;
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function updateStatusChip(state, label) {
  const chip = document.getElementById("statusChip");
  const statusLabel = document.getElementById("statusLabel");
  if (!chip || !statusLabel) return;
  chip.classList.remove("online", "offline", "checking");
  chip.classList.add(state);
  statusLabel.textContent = label;
}

async function checkServerStatus() {
  updateStatusChip("checking", "Checking…");
  try {
    const headers = {};
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/health`, { headers });
    if (!res.ok) throw new Error("unavailable");
    const data = await res.json();
    const healthy = (data.status || "").toLowerCase() === "healthy";
    serverOnline = healthy;
    if (healthy) updateStatusChip("online", "Live");
    else updateStatusChip("offline", "Offline");
  } catch (e) {
    serverOnline = false;
    updateStatusChip("offline", "Offline");
  }
}

function applyTheme(theme) {
  const body = document.body;
  if (!body) return;
  const normalized = theme === "dark" ? "dark" : "light";
  if (normalized === "dark") body.setAttribute("data-theme", "dark");
  else body.removeAttribute("data-theme");
  try {
    localStorage.setItem(THEME_KEY, normalized);
  } catch (e) {
    /* ignore storage errors */
  }
  const toggle = document.getElementById("themeToggle");
  if (toggle) {
    toggle.textContent = normalized === "dark" ? "☀️ Light mode" : "🌙 Dark mode";
  }
}

function toggleTheme() {
  const currentTheme = localStorage.getItem(THEME_KEY) || (document.body.hasAttribute("data-theme") ? "dark" : "light");
  const next = currentTheme === "dark" ? "light" : "dark";
  applyTheme(next);
}

let selectedDocIds = new Set();

async function ingestUrl() {
  const url = document.getElementById("urlInput").value.trim();
  if (!url) return alert("Paste a URL first");
  addMessage(`Ingesting URL: ${url}`, "user");
  addMessage("⏳ Ingesting...", "bot");
  try {
    const res = await fetch(`${API_BASE}/ingest-url`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${getToken()}`
      },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    document.querySelectorAll(".bot")[document.querySelectorAll(".bot").length - 1].remove();
    if (!res.ok) return addMessage(data.detail || "Ingest failed", "bot");
    addMessage(data.summary || `Ingested ${data.chunks_indexed} chunks.`, "bot");
    document.getElementById("urlInput").value = "";
    loadDocs();
  } catch (e) {
    document.querySelectorAll(".bot")[document.querySelectorAll(".bot").length - 1].remove();
    addMessage(`Error: ${e}`, "bot");
  }
}

async function loadDocs() {
  const list = document.getElementById("docsList");
  list.innerHTML = "Loading docs...";
  try {
    const res = await fetch(`${API_BASE}/kb/docs`, {
      headers: {
        "Authorization": `Bearer ${getToken()}`
      }
    });
    const data = await res.json();
    const docs = data.docs || [];
    list.innerHTML = "";
    docs.forEach(doc => {
      const item = document.createElement("div");
      item.className = "doc-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = doc.id;
      cb.checked = selectedDocIds.has(doc.id);
      cb.onchange = (e) => {
        if (e.target.checked) selectedDocIds.add(doc.id);
        else selectedDocIds.delete(doc.id);
      };
      const label = document.createElement("span");
      label.textContent = `${doc.title || doc.id} (${doc.chunks_count || 0})`;
      item.appendChild(cb);
      item.appendChild(label);
      list.appendChild(item);
    });
  } catch (e) {
    list.innerHTML = `Failed to load docs: ${e}`;
  }
}

async function sendMessage() {
  const input = document.getElementById("chatInput");
  const userMessage = input.value.trim();
  if (!userMessage) return;
  addMessage(userMessage, "user");
  input.value = "";
  addMessage("⏳ Thinking...", "bot");

  try {
    const res = await fetch(`${API_BASE}/ask-policy`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${getToken()}`
      },
      body: JSON.stringify({ query: userMessage, doc_ids: Array.from(selectedDocIds) })
    });
    const data = await res.json();
    document.querySelectorAll(".bot")[document.querySelectorAll(".bot").length - 1].remove();
    if (!res.ok) {
      addMessage(data.detail || "Error", "bot");
      return;
    }
    addMessage(data.answer || "No answer.", "bot");
  } catch (e) {
    document.querySelectorAll(".bot")[document.querySelectorAll(".bot").length - 1].remove();
    addMessage(`Error: ${e}`, "bot");
  }
}

/* 🎤 Voice Input */
function startVoiceInput() {
  if (!("webkitSpeechRecognition" in window)) {
    alert("Voice input not supported!");
    return;
  }
  const recognition = new webkitSpeechRecognition();
  recognition.lang = "en-US";
  recognition.start();
  recognition.onresult = (event) => {
    const voiceText = event.results[0][0].transcript;
    document.getElementById("chatInput").value = voiceText;
  };
}

/* 📎 File Upload */
async function sendFile() {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  if (!file) return;

  addMessage(`Uploaded file: ${file.name}`, "user");
  const formData = new FormData();
  formData.append("file", file);
  addMessage("⏳ Extracting and analyzing file...", "bot");

  try {
    const res = await fetch(`${API_BASE}/upload-policy`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${getToken()}`
      },
      body: formData
    });
    const data = await res.json();
    document.querySelectorAll(".bot")[document.querySelectorAll(".bot").length - 1].remove();
    if (!res.ok) {
      addMessage(data.detail || "Upload error", "bot");
      return;
    }
    addMessage(data.summary || "No summary generated.", "bot");
  } catch (e) {
    document.querySelectorAll(".bot")[document.querySelectorAll(".bot").length - 1].remove();
    addMessage(`Error: ${e}`, "bot");
  }
}

// Enter to send
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("chatInput");
  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendMessage();
    });
  }

  const tokenInput = document.getElementById("tokenInput");
  if (tokenInput) tokenInput.value = "";

  const storedTheme = localStorage.getItem(THEME_KEY) || "light";
  applyTheme(storedTheme);

  loadDocs();
  checkServerStatus();
  setInterval(checkServerStatus, 30000);
});
