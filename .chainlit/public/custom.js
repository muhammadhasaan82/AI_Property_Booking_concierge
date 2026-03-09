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
