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
    
    // Current language detection
    let currentLang = 'en';
    const path = window.location.pathname;
    if (path.includes('.zh.')) {
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
            if (currentPath.endsWith('/')) {
                newPath = currentPath + 'index.zh.html';
            } else if (currentPath.endsWith('.html')) {
                newPath = currentPath.replace('.html', '.zh.html');
            } else if (currentPath.endsWith('/index.md')) {
                newPath = currentPath.replace('/index.md', '/index.zh.html');
            } else {
                newPath = currentPath.replace('.md', '.zh.html');
            }
        } else if (currentLang === 'zh' && targetLang === 'en') {
            // Switch from Chinese to English
            newPath = currentPath.replace('.zh.html', '.html');
            if (newPath.endsWith('/index.html')) {
                newPath = newPath.replace('/index.html', '/');
            }
        }
        
        // Check if target file exists, fallback to index if not
        if (newPath) {
            // For development, directly navigate. In production, you might want to check existence
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