const SIDEBAR_STATE_KEY = "chainlit-sidebar-open";
const BRAND_NAME = "AI Booking";

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

function forceHeaderTitle() {
  const titleSelectors = [
    "header .MuiTypography-h6",
    "header h1",
    "header h6",
    ".MuiAppBar-root .MuiTypography-h6",
    ".MuiAppBar-root h1",
    ".MuiAppBar-root h6",
  ];

  document.querySelectorAll(titleSelectors.join(", ")).forEach((node) => {
    const text = (node.textContent || "").trim();
    if (/chainlit|ai booking concierge/i.test(text)) {
      node.textContent = BRAND_NAME;
    }
  });

  if (/chainlit|ai booking concierge/i.test(document.title || "")) {
    document.title = document.title.replace(/chainlit|ai booking concierge/gi, BRAND_NAME);
  }
}

window.addEventListener("load", () => {
  let attempts = 0;
  const interval = window.setInterval(() => {
    attempts += 1;
    const opened = ensureSidebarOpen();
    forceHeaderTitle();
    if (opened || attempts >= 20) {
      window.clearInterval(interval);
    }
  }, 500);

  const observer = new MutationObserver(() => {
    ensureSidebarOpen();
    forceHeaderTitle();
  });

  observer.observe(document.body, { childList: true, subtree: true });
  window.setTimeout(() => observer.disconnect(), 15000);

  forceHeaderTitle();
});
