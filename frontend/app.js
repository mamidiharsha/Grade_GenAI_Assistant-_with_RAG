/**
 * GenAI Assistant — Frontend Application
 * Handles chat interaction, session management, and message rendering.
 */

// ─── Configuration ──────────────────────────────────────────────
const API_BASE = window.location.origin;
const API_CHAT = `${API_BASE}/api/chat`;
const API_HEALTH = `${API_BASE}/health`;
const SESSION_KEY = "genai_session_id";
const MESSAGES_KEY = "genai_messages";

// ─── State ──────────────────────────────────────────────────────
let sessionId = getOrCreateSessionId();
let isLoading = false;
let totalTokensUsed = 0;

// ─── DOM Elements ───────────────────────────────────────────────
const chatMessages = document.getElementById("chatMessages");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const welcomeSection = document.getElementById("welcomeSection");
const tokenCounter = document.getElementById("tokenCounter");
const statusIndicator = document.getElementById("statusIndicator");

// ─── Session Management ────────────────────────────────────────
function getOrCreateSessionId() {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
        id = generateUUID();
        localStorage.setItem(SESSION_KEY, id);
    }
    return id;
}

function generateUUID() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

function startNewChat() {
    sessionId = generateUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
    localStorage.removeItem(MESSAGES_KEY);
    totalTokensUsed = 0;
    updateTokenCounter();

    // Clear messages and show welcome
    chatMessages.innerHTML = "";
    chatMessages.appendChild(createWelcomeSection());
    welcomeSection && (welcomeSection.style.display = "");
    bindChipListeners();
}

// ─── Message Persistence ────────────────────────────────────────
function saveMessages() {
    const messages = [];
    chatMessages.querySelectorAll(".message").forEach((el) => {
        const isUser = el.classList.contains("message-user");
        const text = el.querySelector(".message-bubble").innerText;
        const time = el.querySelector(".message-time")?.textContent || "";
        const tokens = el.dataset.tokens || "0";
        messages.push({ role: isUser ? "user" : "assistant", text, time, tokens });
    });
    localStorage.setItem(MESSAGES_KEY, JSON.stringify(messages));
}

function loadMessages() {
    const saved = localStorage.getItem(MESSAGES_KEY);
    if (!saved) return;

    try {
        const messages = JSON.parse(saved);
        if (messages.length === 0) return;

        // Hide welcome section
        const welcome = document.getElementById("welcomeSection");
        if (welcome) welcome.remove();

        messages.forEach((msg) => {
            if (msg.role === "user") {
                appendMessage(msg.text, "user", msg.time, false);
            } else {
                appendMessage(msg.text, "assistant", msg.time, false, parseInt(msg.tokens) || 0);
            }
        });

        scrollToBottom();
    } catch (e) {
        console.error("Failed to load messages:", e);
    }
}

// ─── Message Rendering ─────────────────────────────────────────
function appendMessage(text, role, timestamp, animate = true, tokens = 0) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message message-${role}`;
    if (tokens) messageDiv.dataset.tokens = tokens;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = role === "assistant" ? renderMarkdown(text) : escapeHtml(text);

    const meta = document.createElement("div");
    meta.className = "message-meta";

    const timeSpan = document.createElement("span");
    timeSpan.className = "message-time";
    timeSpan.textContent = timestamp || formatTime(new Date());
    meta.appendChild(timeSpan);

    if (role === "assistant" && tokens > 0) {
        const tokenSpan = document.createElement("span");
        tokenSpan.className = "message-tokens";
        tokenSpan.textContent = `${tokens} tokens`;
        meta.appendChild(tokenSpan);
    }

    messageDiv.appendChild(bubble);
    messageDiv.appendChild(meta);

    if (!animate) {
        messageDiv.style.animation = "none";
    }

    chatMessages.appendChild(messageDiv);

    if (animate) {
        scrollToBottom();
        saveMessages();
    }
}

function showTypingIndicator() {
    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    typing.id = "typingIndicator";
    typing.innerHTML = `
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        <span class="typing-text">Searching knowledge base...</span>
    `;
    chatMessages.appendChild(typing);
    scrollToBottom();
}

function removeTypingIndicator() {
    const typing = document.getElementById("typingIndicator");
    if (typing) typing.remove();
}

// ─── Markdown Rendering ────────────────────────────────────────
function renderMarkdown(text) {
    let html = escapeHtml(text);

    // Code blocks (```...```)
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Unordered lists
    html = html.replace(/^[\s]*[-•]\s+(.+)/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Ordered lists
    html = html.replace(/^[\s]*\d+\.\s+(.+)/gm, "<li>$1</li>");

    // Line breaks to paragraphs
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");

    // Wrap in paragraph if not already wrapped
    if (!html.startsWith("<")) {
        html = `<p>${html}</p>`;
    }

    return html;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ─── API Communication ─────────────────────────────────────────
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Hide welcome section
    const welcome = document.getElementById("welcomeSection");
    if (welcome) welcome.remove();

    // Show user message
    appendMessage(message, "user");
    messageInput.value = "";
    autoResizeTextarea();
    updateSendButton();

    // Show loading state
    isLoading = true;
    sendBtn.disabled = true;
    showTypingIndicator();

    try {
        const response = await fetch(API_CHAT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sessionId: sessionId,
                message: message,
            }),
        });

        removeTypingIndicator();

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMsg = errorData.detail || errorData.error || `Error: ${response.status}`;
            appendMessage(errorMsg, "assistant", null, true, 0);
            return;
        }

        const data = await response.json();

        // Update token counter
        totalTokensUsed += data.tokensUsed || 0;
        updateTokenCounter();

        // Show assistant response
        appendMessage(data.reply, "assistant", null, true, data.tokensUsed || 0);

    } catch (error) {
        removeTypingIndicator();
        console.error("Chat error:", error);
        appendMessage(
            "Unable to connect to the server. Please check if the backend is running and try again.",
            "assistant",
            null,
            true,
            0
        );
    } finally {
        isLoading = false;
        updateSendButton();
    }
}

// ─── Health Check ───────────────────────────────────────────────
async function checkHealth() {
    try {
        const response = await fetch(API_HEALTH);
        if (response.ok) {
            setStatus("Online", true);
        } else {
            setStatus("Offline", false);
        }
    } catch {
        setStatus("Offline", false);
    }
}

function setStatus(text, online) {
    const dot = statusIndicator.querySelector(".status-dot");
    const label = statusIndicator.querySelector(".status-text");
    label.textContent = text;
    if (online) {
        dot.style.background = "var(--success)";
        statusIndicator.style.borderColor = "rgba(16, 185, 129, 0.2)";
        statusIndicator.style.background = "rgba(16, 185, 129, 0.1)";
        label.style.color = "var(--success)";
    } else {
        dot.style.background = "var(--error)";
        dot.style.animation = "none";
        statusIndicator.style.borderColor = "rgba(239, 68, 68, 0.2)";
        statusIndicator.style.background = "rgba(239, 68, 68, 0.1)";
        label.style.color = "var(--error)";
    }
}

// ─── Utility Functions ──────────────────────────────────────────
function formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isLoading;
}

function updateTokenCounter() {
    if (totalTokensUsed > 0) {
        tokenCounter.textContent = `Total: ${totalTokensUsed.toLocaleString()} tokens`;
    } else {
        tokenCounter.textContent = "";
    }
}

function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

function createWelcomeSection() {
    const section = document.createElement("div");
    section.className = "welcome-section";
    section.id = "welcomeSection";
    section.innerHTML = `
        <div class="welcome-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
            </svg>
        </div>
        <h2>Welcome to GenAI Assistant</h2>
        <p>I'm powered by Retrieval-Augmented Generation (RAG) to provide accurate, grounded answers from our knowledge base.</p>
        <div class="suggestion-chips">
            <button class="chip" data-query="How can I reset my password?">🔑 Reset password</button>
            <button class="chip" data-query="What subscription plans are available?">💳 Subscription plans</button>
            <button class="chip" data-query="How do I enable two-factor authentication?">🔒 Enable 2FA</button>
            <button class="chip" data-query="How can I export my data?">📦 Data export</button>
        </div>
    `;
    return section;
}

// ─── Chip Listeners ─────────────────────────────────────────────
function bindChipListeners() {
    document.querySelectorAll(".chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            const query = chip.dataset.query;
            if (query) {
                messageInput.value = query;
                autoResizeTextarea();
                updateSendButton();
                sendMessage();
            }
        });
    });
}

// ─── Event Listeners ────────────────────────────────────────────
sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

messageInput.addEventListener("input", () => {
    updateSendButton();
    autoResizeTextarea();
});

newChatBtn.addEventListener("click", startNewChat);

// ─── Initialization ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadMessages();
    bindChipListeners();
    checkHealth();

    // Periodic health check
    setInterval(checkHealth, 30000);

    // Focus input
    messageInput.focus();
});
