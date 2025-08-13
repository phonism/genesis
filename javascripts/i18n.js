/**
 * Genesis Documentation Language Switcher
 * Switches between English and Chinese versions of documentation
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
    
    // Current language detection based on URL
    let currentLang = 'en';
    const path = window.location.pathname;
    // Check if we're on a Chinese page (ends with .zh/ or contains .zh.)
    if (path.includes('.zh/') || path.includes('/zh/') || path.endsWith('.zh/')) {
        currentLang = 'zh';
    }
    
    // Create language switcher
    function createLanguageSwitcher() {
        const header = document.querySelector('.md-header__inner');
        if (!header) return;
        
        const switcher = document.createElement('div');
        switcher.className = 'md-header__language-switcher';
        switcher.innerHTML = `
            <button class="md-header__button md-icon" id="lang-switcher" title="Switch Language">
                <span class="lang-icon">${languages[currentLang].icon}</span>
                <span class="lang-text">${languages[currentLang].flag}</span>
            </button>
            <div class="lang-dropdown" id="lang-dropdown">
                <a href="#" data-lang="en" class="lang-option ${currentLang === 'en' ? 'active' : ''}">
                    ${languages.en.icon} ${languages.en.name}
                </a>
                <a href="#" data-lang="zh" class="lang-option ${currentLang === 'zh' ? 'active' : ''}">
                    ${languages.zh.icon} ${languages.zh.name}
                </a>
            </div>
        `;
        
        // Insert before search button
        const searchButton = header.querySelector('[data-md-component="search"]');
        if (searchButton) {
            header.insertBefore(switcher, searchButton);
        } else {
            header.appendChild(switcher);
        }
        
        // Add event listeners
        const button = switcher.querySelector('#lang-switcher');
        const dropdown = switcher.querySelector('#lang-dropdown');
        
        button.addEventListener('click', (e) => {
            e.preventDefault();
            dropdown.classList.toggle('show');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!switcher.contains(e.target)) {
                dropdown.classList.remove('show');
            }
        });
        
        // Language switch handlers
        switcher.querySelectorAll('.lang-option').forEach(option => {
            option.addEventListener('click', (e) => {
                e.preventDefault();
                const targetLang = option.getAttribute('data-lang');
                switchLanguage(targetLang);
            });
        });
    }
    
    // Switch language function
    function switchLanguage(targetLang) {
        if (targetLang === currentLang) return;
        
        const currentPath = window.location.pathname;
        let newPath;
        
        if (currentLang === 'en' && targetLang === 'zh') {
            // Switch from English to Chinese
            // MkDocs converts file.zh.md to file.zh/ directory structure
            if (currentPath === '/' || currentPath.endsWith('/genesis/')) {
                // Root index page
                newPath = currentPath + 'index.zh/';
            } else if (currentPath.endsWith('/')) {
                // Directory pages like /getting-started/
                newPath = currentPath + 'index.zh/';
            } else {
                // Normal pages - add .zh before the final /
                newPath = currentPath.replace(/\/$/, '.zh/');
            }
        } else if (currentLang === 'zh' && targetLang === 'en') {
            // Switch from Chinese to English
            if (currentPath.includes('index.zh/')) {
                // Handle index.zh/ -> / or directory/
                newPath = currentPath.replace('/index.zh/', '/');
            } else {
                // Handle page.zh/ -> page/
                newPath = currentPath.replace('.zh/', '/');
            }
            
            // Clean up double slashes
            newPath = newPath.replace('//', '/');
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
                position: relative;
                display: flex;
                align-items: center;
                margin-right: 0.5rem;
            }
            
            .md-header__language-switcher button {
                background: none;
                border: none;
                color: var(--md-primary-fg-color--light);
                cursor: pointer;
                padding: 0.5rem;
                border-radius: 0.25rem;
                display: flex;
                align-items: center;
                gap: 0.25rem;
                transition: background-color 0.2s;
            }
            
            .md-header__language-switcher button:hover {
                background-color: var(--md-accent-fg-color--transparent);
            }
            
            .lang-icon {
                font-size: 1.1rem;
            }
            
            .lang-text {
                font-size: 0.75rem;
                font-weight: 600;
            }
            
            .lang-dropdown {
                position: absolute;
                top: 100%;
                right: 0;
                background: var(--md-default-bg-color);
                border: 1px solid var(--md-default-fg-color--lightest);
                border-radius: 0.25rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                display: none;
                z-index: 1000;
                min-width: 120px;
            }
            
            .lang-dropdown.show {
                display: block;
            }
            
            .lang-option {
                display: block;
                padding: 0.5rem 0.75rem;
                color: var(--md-default-fg-color);
                text-decoration: none;
                font-size: 0.875rem;
                transition: background-color 0.2s;
            }
            
            .lang-option:hover {
                background-color: var(--md-default-fg-color--lightest);
            }
            
            .lang-option.active {
                background-color: var(--md-accent-fg-color--transparent);
                color: var(--md-accent-fg-color);
            }
        `;
        document.head.appendChild(style);
    }
    
    // Initialize when DOM is ready
    function init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                addStyles();
                createLanguageSwitcher();
            });
        } else {
            addStyles();
            createLanguageSwitcher();
        }
    }
    
    init();
})();