// Custom JS to open the sidebar/thread panel by default
window.addEventListener('load', function() {
    setTimeout(function() {
        const sidebarToggle = document.querySelector('[data-testid="sidebar-toggle"], .sidebar-toggle, button[aria-label*="sidebar"], button[aria-label*="history"]');
        const threadButton = document.querySelector('button[aria-label="Toggle thread history"], button[aria-label="thread history"]');
        
        if (sidebarToggle) {
            sidebarToggle.click();
        } else if (threadButton) {
            threadButton.click();
        }
        
        try {
            localStorage.setItem('chainlit-sidebar-open', 'true');
        } catch (e) {}
    }, 1000);
});
