const config = window.APP_CONFIG || {};
const storageKey = "ai_property_booking_api_base_url";
let sessionId = window.crypto?.randomUUID ? window.crypto.randomUUID() : String(Date.now());
const userId = "pages_user";

const apiInput = document.querySelector("#apiBaseUrl");
const saveConfigButton = document.querySelector("#saveConfigButton");
const healthButton = document.querySelector("#healthButton");
const healthOutput = document.querySelector("#healthOutput");
const statusBadge = document.querySelector("#statusBadge");
const configNotice = document.querySelector("#configNotice");
const docsLink = document.querySelector("#docsLink");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");

function cleanBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function getApiBaseUrl() {
  return cleanBaseUrl(localStorage.getItem(storageKey) || config.API_BASE_URL || "");
}

function updateConfigState() {
  const apiBaseUrl = getApiBaseUrl();
  apiInput.value = apiBaseUrl;
  const configured = Boolean(apiBaseUrl);
  configNotice.hidden = configured;
  docsLink.href = configured ? `${apiBaseUrl}/docs` : "#";
  docsLink.classList.toggle("disabled", !configured);
  docsLink.setAttribute("aria-disabled", configured ? "false" : "true");
  messageInput.disabled = !configured;
  chatForm.querySelector("button").disabled = !configured;
  if (!configured) {
    setBadge("Not configured", "warning");
  }
}

function setBadge(text, state) {
  statusBadge.textContent = text;
  statusBadge.className = `badge badge-${state}`;
}

function printJson(target, value) {
  target.textContent = JSON.stringify(value, null, 2);
}

function addMessage(role, text) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = text;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

async function checkHealth() {
  const apiBaseUrl = getApiBaseUrl();
  if (!apiBaseUrl) {
    setBadge("Not configured", "warning");
    healthOutput.textContent = "Backend API is not configured.";
    return;
  }

  setBadge("Checking", "warning");
  healthOutput.textContent = `GET ${apiBaseUrl}/health`;

  try {
    const response = await fetch(`${apiBaseUrl}/health`, {
      method: "GET",
      headers: { Accept: "application/json" }
    });
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    setBadge(response.ok ? "Online" : "Error", response.ok ? "success" : "danger");
    printJson(healthOutput, { status: response.status, body });
  } catch (error) {
    setBadge("Unavailable", "danger");
    printJson(healthOutput, {
      error: "Could not reach backend API. Check the URL, HTTPS, and backend CORS settings.",
      detail: String(error)
    });
  }
}

async function sendChatMessage(message) {
  const apiBaseUrl = getApiBaseUrl();
  addMessage("user", message);

  try {
    const response = await fetch(`${apiBaseUrl}/chat/message`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      body: JSON.stringify({
        message,
        user_id: userId,
        session_id: sessionId
      })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    sessionId = payload.session_id || sessionId;
    addMessage("assistant", payload.reply || "No reply returned.");
  } catch (error) {
    addMessage(
      "assistant",
      `I could not reach the backend chat API. ${String(error)}`
    );
  }
}

saveConfigButton.addEventListener("click", () => {
  const value = cleanBaseUrl(apiInput.value);
  if (value) {
    localStorage.setItem(storageKey, value);
  } else {
    localStorage.removeItem(storageKey);
  }
  updateConfigState();
});

healthButton.addEventListener("click", checkHealth);

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }
  messageInput.value = "";
  await sendChatMessage(message);
});

updateConfigState();
