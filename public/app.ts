// ======= Integrate Chatbot API to Frontend =======

const logEl = document.getElementById("log") as HTMLElement;
const chatInput = document.getElementById("chatInput") as HTMLInputElement;
const chatSendBtn = document.getElementById("chatSendBtn") as HTMLButtonElement;

function logChat(line: string) {
  const ts = new Date().toLocaleTimeString();
  logEl.textContent += `[${ts}] ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

if (chatSendBtn && chatInput) {
  chatSendBtn.onclick = async () => {
    const msg = chatInput.value.trim();
    if (!msg) return;
    logChat(`You: ${msg}`);
    chatInput.value = "";
    try {
      const res = await fetch("/api/v1/chat/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      logChat(`Assistant: ${data.reply || JSON.stringify(data)}`);
    } catch (e: any) {
      logChat(`Error: ${e.message || e}`);
    }
  };
  chatInput.onkeydown = (e) => {
    if (e.key === "Enter") chatSendBtn.onclick!(null);
  };
}
