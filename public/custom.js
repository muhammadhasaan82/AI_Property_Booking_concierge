const SIDEBAR_STATE_KEY = "chainlit-sidebar-open";

function findSidebarToggle() {
  return document.querySelector(
    [
      '[data-testid="sidebar-toggle"]',
      'button[aria-label*="Open sidebar"]',
      'button[aria-label*="Toggle sidebar"]',
      'button[aria-label*="thread history"]',
      'button[aria-label*="history"]',
    ].join(", ")
  );
}

function sidebarLooksClosed(toggle) {
  const label = (toggle.getAttribute("aria-label") || "").toLowerCase();
  const expanded = toggle.getAttribute("aria-expanded");
  return expanded === "false" || label.includes("open");
}

function ensureSidebarOpen() {
  const toggle = findSidebarToggle();
  if (!toggle) {
    return false;
  }

  if (sidebarLooksClosed(toggle)) {
    toggle.click();
  }

  try {
    localStorage.setItem(SIDEBAR_STATE_KEY, "true");
  } catch (error) {}

  return true;
}

window.addEventListener("load", () => {
  let attempts = 0;
  const interval = window.setInterval(() => {
    attempts += 1;
    const opened = ensureSidebarOpen();
    if (opened || attempts >= 20) {
      window.clearInterval(interval);
    }
  }, 500);

  const observer = new MutationObserver(() => {
    ensureSidebarOpen();
  });

  observer.observe(document.body, { childList: true, subtree: true });
  window.setTimeout(() => observer.disconnect(), 15000);
});
