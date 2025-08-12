# Genesis Documentation

This directory contains the documentation for the Genesis Deep Learning Framework.

## 📚 Documentation Structure

```
docs/
├── api-reference/      # API documentation
├── architecture/       # Architecture design docs
├── core-components/    # Core system documentation
├── getting-started/    # Tutorials and guides
├── models/            # Model implementations
├── javascripts/       # Custom JavaScript
├── stylesheets/       # Custom CSS
├── requirements.txt   # Python dependencies
└── mkdocs.yml        # MkDocs configuration (in parent directory)
```

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Serve locally:**
   ```bash
   # From project root
   mkdocs serve
   # Or use our script
   ./build_docs.sh
   ```

3. **Build static site:**
   ```bash
   mkdocs build
   # Or
   ./build_docs.sh build
   ```

### Deploy to GitHub Pages

#### Method 1: Using mkdocs gh-deploy (Recommended)
```bash
# This creates/updates the gh-pages branch automatically
mkdocs gh-deploy --force
# Or use our script
./deploy_docs.sh
```

#### Method 2: Using GitHub Actions
The repository includes a GitHub Actions workflow that automatically deploys documentation when changes are pushed to the main branch.

1. Go to repository **Settings** → **Pages**
2. Under **Build and deployment**, select:
   - Source: **Deploy from a branch**
   - Branch: **gh-pages** / **/ (root)**
3. Click **Save**

The documentation will be available at: `https://YOUR_USERNAME.github.io/genesis/`

## 📝 Writing Documentation

### Markdown Files
- All documentation is written in Markdown
- Files should use `.md` extension
- Follow the existing structure and naming conventions

### Adding New Pages
1. Create a new `.md` file in the appropriate directory
2. Update `mkdocs.yml` to include the new page in navigation
3. Test locally with `mkdocs serve`

### Code Examples
Use fenced code blocks with language specification:
````markdown
```python
import genesis
import genesis.nn as nn

model = nn.Linear(10, 5)
```
````

### Mathematical Expressions
Use LaTeX syntax for math:
```markdown
Inline math: $f(x) = x^2$
Block math:
$$
\nabla_\theta L = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(f_\theta(x_i), y_i)
$$
```

### Admonitions
Use admonitions for notes, warnings, etc:
```markdown
!!! note "Important"
    This is an important note.

!!! warning
    This is a warning message.

!!! tip
    This is a helpful tip.
```

## 🛠️ Configuration

The documentation uses [MkDocs](https://www.mkdocs.org/) with the [Material](https://squidfunk.github.io/mkdocs-material/) theme.

Key configuration files:
- `mkdocs.yml` - Main configuration (in project root)
- `requirements.txt` - Python dependencies
- `stylesheets/extra.css` - Custom CSS
- `javascripts/mathjax.js` - MathJax configuration

## 📦 Dependencies

Main dependencies:
- `mkdocs` - Static site generator
- `mkdocs-material` - Material theme
- `mkdocs-minify-plugin` - Minification
- `mkdocs-git-revision-date-localized-plugin` - Git revision dates
- `pymdown-extensions` - Markdown extensions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `mkdocs serve`
5. Submit a pull request

## 📄 License

The documentation is licensed under the same license as the Genesis project.