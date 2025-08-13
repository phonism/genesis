/**
 * Genesis Documentation Language Switcher
 * Matches actual MkDocs URL structure
 */

(function() {
    'use strict';
    
    // Language configuration
    const languages = {
        'en': {
            name: 'English',
            icon: '🇺🇸',
            flag: 'EN'
        },
        'zh': {
            name: '中文',
            icon: '🇨🇳', 
            flag: '中'
        }
    };
    
    // Detect current language based on URL
    let currentLang = 'en';
    const path = window.location.pathname;
    if (path.includes('/zh/') || path.includes('.zh/')) {
        currentLang = 'zh';
    }
    
    // Create language switcher
    function createLanguageSwitcher() {
        const header = document.querySelector('.md-header__inner');
        if (!header) return;
        
        const switcher = document.createElement('div');
        switcher.className = 'md-header__language-switcher';
        
        // Create both language options, highlight current one
        switcher.innerHTML = `
            <button class="lang-option ${currentLang === 'en' ? 'active' : ''}" data-lang="en" title="Switch to English">
                <span class="lang-icon">${languages.en.icon}</span>
                <span class="lang-text">${languages.en.flag}</span>
            </button>
            <span class="lang-separator">|</span>
            <button class="lang-option ${currentLang === 'zh' ? 'active' : ''}" data-lang="zh" title="Switch to Chinese">
                <span class="lang-icon">${languages.zh.icon}</span>
                <span class="lang-text">${languages.zh.flag}</span>
            </button>
        `;
        
        // Insert before search button
        const searchButton = header.querySelector('[data-md-component="search"]');
        if (searchButton) {
            header.insertBefore(switcher, searchButton);
        } else {
            header.appendChild(switcher);
        }
        
        // Add event listeners
        switcher.querySelectorAll('.lang-option').forEach(option => {
            option.addEventListener('click', (e) => {
                e.preventDefault();
                const targetLang = option.getAttribute('data-lang');
                if (targetLang !== currentLang) {
                    switchLanguage(targetLang);
                }
            });
        });
    }
    
    // Switch language function that matches MkDocs structure
    function switchLanguage(targetLang) {
        if (targetLang === currentLang) return;
        
        const currentPath = window.location.pathname;
        let newPath;
        
        if (currentLang === 'en' && targetLang === 'zh') {
            // English to Chinese conversion
            if (currentPath === '/' || currentPath.endsWith('/genesis/')) {
                // Root page: /genesis/ -> /genesis/zh/
                newPath = currentPath.replace(/\/genesis\/$/, '/genesis/zh/') || '/genesis/zh/';
            } else if (currentPath.includes('/genesis/')) {
                // Other pages: /genesis/xxx/ -> /genesis/zh/xxx.zh/
                newPath = currentPath.replace('/genesis/', '/genesis/zh/').replace(/\/$/, '.zh/');
            } else {
                // Fallback
                newPath = '/genesis/zh/';
            }
        } else if (currentLang === 'zh' && targetLang === 'en') {
            // Chinese to English conversion
            if (currentPath.includes('/genesis/zh/')) {
                if (currentPath === '/genesis/zh/' || currentPath.endsWith('/genesis/zh/index.zh/')) {
                    // Root Chinese page -> Root English page
                    newPath = '/genesis/';
                } else {
                    // Other Chinese pages: /genesis/zh/xxx.zh/ -> /genesis/xxx/
                    newPath = currentPath.replace('/genesis/zh/', '/genesis/').replace('.zh/', '/');
                }
            } else {
                // Fallback
                newPath = '/genesis/';
            }
        }
        
        // Navigate to new path
        if (newPath) {
            window.location.href = newPath;
        }
    }
    
    // Add CSS styles
    function addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .md-header__language-switcher {
                display: flex;
                align-items: center;
                margin-right: 0.5rem;
                gap: 0.25rem;
            }
            
            .lang-option {
                background: none;
                border: none;
                color: var(--md-primary-fg-color--light);
                cursor: pointer;
                padding: 0.375rem 0.5rem;
                border-radius: 0.25rem;
                display: flex;
                align-items: center;
                gap: 0.25rem;
                transition: all 0.2s ease;
                opacity: 0.7;
                font-size: 0.875rem;
            }
            
            .lang-option:hover {
                background-color: var(--md-accent-fg-color--transparent);
                opacity: 1;
                transform: translateY(-1px);
            }
            
            .lang-option.active {
                background-color: var(--md-accent-fg-color--transparent);
                color: var(--md-accent-fg-color);
                opacity: 1;
                font-weight: 600;
            }
            
            .lang-icon {
                font-size: 1rem;
            }
            
            .lang-text {
                font-size: 0.75rem;
                font-weight: inherit;
            }
            
            .lang-separator {
                color: var(--md-primary-fg-color--light);
                opacity: 0.5;
                font-size: 0.875rem;
                margin: 0 0.125rem;
            }
            
            /* Responsive design */
            @media screen and (max-width: 768px) {
                .lang-text {
                    display: none;
                }
                
                .lang-option {
                    padding: 0.375rem;
                }
                
                .lang-separator {
                    margin: 0 0.25rem;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Initialize function
    function initSwitcher() {
        // Remove existing switcher if present
        const existingSwitcher = document.querySelector('.md-header__language-switcher');
        if (existingSwitcher) {
            existingSwitcher.remove();
        }
        
        // Re-detect current language
        const path = window.location.pathname;
        currentLang = (path.includes('/zh/') || path.includes('.zh/')) ? 'zh' : 'en';
        
        // Create new switcher
        createLanguageSwitcher();
    }
    
    // Initialize when DOM is ready
    function init() {
        addStyles();
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initSwitcher);
        } else {
            initSwitcher();
        }
        
        // Listen for MkDocs instant navigation events
        document.addEventListener('DOMContentLoaded', () => {
            // MkDocs Material instant navigation
            if (typeof app !== 'undefined' && app.document$) {
                app.document$.subscribe(initSwitcher);
            }
        });
        
        // Fallback for page navigation detection
        let currentUrl = window.location.href;
        setInterval(() => {
            if (window.location.href !== currentUrl) {
                currentUrl = window.location.href;
                setTimeout(initSwitcher, 100); // Small delay to ensure DOM is updated
            }
        }, 500);
    }
    
    init();
})();