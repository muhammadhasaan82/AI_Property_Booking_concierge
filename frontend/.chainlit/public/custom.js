// Custom JS to open the sidebar/thread panel by default
window.addEventListener('load', function() {
    // Wait for Chainlit UI to fully load
    setTimeout(function() {
        // Try to find and click the sidebar toggle button to open it
        const sidebarToggle = document.querySelector('[data-testid="sidebar-toggle"], .sidebar-toggle, button[aria-label*="sidebar"], button[aria-label*="history"]');
        
        // Alternative: look for thread history button
        const threadButton = document.querySelector('button[aria-label="Toggle thread history"], button[aria-label="thread history"]');
        
        if (sidebarToggle) {
            sidebarToggle.click();
        } else if (threadButton) {
            threadButton.click();
        }
        
        // Also try to ensure the sidebar/threads panel is visible by checking localStorage
        // Chainlit stores sidebar state, so we can force it open
        try {
            localStorage.setItem('chainlit-sidebar-open', 'true');
        } catch (e) {
            // Ignore localStorage errors
        }
    }, 1000); // Wait 1 second for UI to initialize
});

// Replace "Chainlit" with "AI Booking Concierge" on login page
const BRAND_TOKEN = 'Chainlit';
const BRAND_NAME = 'AI Booking Concierge';

function replaceBrandText() {
    if (!document.body) return;

    // Replace text nodes containing "Chainlit"
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
    let node;
    while ((node = walker.nextNode())) {
        if (node.nodeValue && node.nodeValue.includes(BRAND_TOKEN)) {
            node.nodeValue = node.nodeValue.split(BRAND_TOKEN).join(BRAND_NAME);
        }
    }

    // Replace common attributes
    const attrTargets = document.querySelectorAll('[aria-label], [title]');
    attrTargets.forEach(el => {
        ['aria-label', 'title'].forEach(attr => {
            const value = el.getAttribute(attr);
            if (value && value.includes(BRAND_TOKEN)) {
                el.setAttribute(attr, value.split(BRAND_TOKEN).join(BRAND_NAME));
            }
        });
    });

    // Replace document title if needed
    if (document.title && document.title.includes(BRAND_TOKEN)) {
        document.title = document.title.split(BRAND_TOKEN).join(BRAND_NAME);
    }
}

function ensureLogoText() {
    const logoCandidates = document.querySelectorAll('img, svg');
    logoCandidates.forEach(logo => {
        const src = logo.getAttribute('src') || '';
        const aria = logo.getAttribute('aria-label') || '';
        const looksLikeLogo = src.toLowerCase().includes('chainlit') ||
            src.toLowerCase().includes('logo') ||
            aria.toLowerCase().includes('chainlit');

        if (!looksLikeLogo) return;

        const parent = logo.parentElement;
        if (!parent) return;

        const parentText = (parent.textContent || '').trim();
        if (parentText.includes(BRAND_NAME)) return;

        if (!parentText || parentText === BRAND_TOKEN) {
            let label = parent.querySelector('[data-brand-name="true"]');
            if (!label) {
                label = document.createElement('span');
                label.setAttribute('data-brand-name', 'true');
                label.style.fontSize = '24px';
                label.style.fontWeight = '700';
                label.style.color = '#F80061';
                label.style.marginLeft = '10px';
                label.style.fontFamily = 'inherit';
                parent.appendChild(label);
            }
            label.textContent = BRAND_NAME;
        }
    });
}

setInterval(function() {
    replaceBrandText();
    ensureLogoText();
}, 500);
