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
setInterval(function() {
    const logoImgs = document.querySelectorAll('img');
    logoImgs.forEach(img => {
        if (img.src && (img.src.includes('logo') || img.src.includes('chainlit'))) {
            const parent = img.parentElement;
            if (parent && !parent.hasAttribute('data-replaced-title')) {
                const span = document.createElement('span');
                span.textContent = 'AI Booking Concierge';
                span.style.fontSize = '24px';
                span.style.fontWeight = 'bold';
                span.style.color = '#F80061';
                span.style.fontFamily = 'Inter, sans-serif';
                span.style.marginLeft = '10px';
                
                img.style.display = 'none';
                parent.appendChild(span);
                parent.setAttribute('data-replaced-title', 'true');
            }
        }
    });

    // Replace all text nodes containing "Chainlit" anywhere in the DOM
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
    let node;
    while ((node = walker.nextNode())) {
        if (node.nodeValue && node.nodeValue.trim() === 'Chainlit') {
            node.nodeValue = node.nodeValue.replace('Chainlit', 'AI Booking Concierge');
        }
    }
}, 500);
