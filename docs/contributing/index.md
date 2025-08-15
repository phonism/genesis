# Contributing Guide

Welcome to contribute code to the Genesis deep learning framework! This guide will help you understand how to participate in project development.

## 🤝 Ways to Contribute

### Code Contributions
- Fix bugs
- Add new features
- Performance optimization
- Test improvements

### Documentation Contributions
- Improve existing documentation
- Add tutorials and examples
- Translate documentation
- API documentation improvements

### Community Contributions
- Answer questions
- Code review
- Issue reporting
- Feature suggestions

## 📋 Development Workflow

### 1. Preparation

```bash
# Fork the project to your GitHub account
# Clone your fork
git clone https://github.com/YOUR_USERNAME/genesis.git
cd genesis

# Add upstream repository
git remote add upstream https://github.com/phonism/genesis.git

# Create development branch
git checkout -b feature/your-feature-name
```

### 2. Development Environment Setup

See the [Development Environment Configuration](development.md) documentation for details.

### 3. Code Development

- Follow coding standards
- Add unit tests
- Update relevant documentation
- Write clear commit messages

### 4. Testing and Verification

```bash
# Run test suite
python -m pytest tests/ -v

# Run code format checks
black genesis/ tests/
flake8 genesis/ tests/

# Run type checking
mypy genesis/
```

### 5. Submit Pull Request

- Ensure all tests pass
- Write detailed PR description
- Link related issues
- Wait for code review

## 📝 Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) standards:

```python
# Good example
def compute_attention_weights(query, key, scale_factor):
    """Compute scaled dot-product attention weights.
    
    Args:
        query: Query tensor of shape [batch, seq_len, hidden_dim]
        key: Key tensor of shape [batch, seq_len, hidden_dim] 
        scale_factor: Scaling factor for attention scores
        
    Returns:
        Attention weights of shape [batch, seq_len, seq_len]
    """
    scores = genesis.matmul(query, key.transpose(-2, -1))
    scaled_scores = scores * scale_factor
    return genesis.softmax(scaled_scores, dim=-1)
```

### Documentation Strings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """One line summary of the function.
    
    More detailed description if needed. Can span multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return param1 > 0
```

### Writing Tests

```python
import pytest
import genesis

class TestAttention:
    """Test attention mechanisms."""
    
    def test_basic_attention(self):
        """Test basic attention computation."""
        batch_size, seq_len, hidden_dim = 2, 4, 8
        
        query = genesis.randn(batch_size, seq_len, hidden_dim)
        key = genesis.randn(batch_size, seq_len, hidden_dim)
        value = genesis.randn(batch_size, seq_len, hidden_dim)
        
        attention = genesis.nn.MultiHeadAttention(hidden_dim, num_heads=2)
        output = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        
    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_different_head_counts(self, num_heads):
        """Test attention with different head counts."""
        # Test implementation
        pass
```

## 🚀 Development Best Practices

### 1. Branch Management

```bash
# Main branches
main          # Stable version
develop       # Development version

# Feature branches
feature/xxx   # New feature development
bugfix/xxx    # Bug fixes
hotfix/xxx    # Hotfixes
```

### 2. Commit Message Format

```
type(scope): brief description

Detailed description (optional)

Fixes #123
```

Type descriptions:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation updates
- `style`: Code formatting adjustments
- `refactor`: Refactoring
- `perf`: Performance optimization
- `test`: Test-related
- `chore`: Build tools, etc.

### 3. Performance Considerations

- Avoid unnecessary memory copies
- Use in-place operations when possible
- Consider CUDA kernel memory access patterns
- Add performance benchmarks

## 🐛 Bug Reports

When submitting bugs, please include:

1. **Environment Information**
   - Genesis version
   - Python version
   - CUDA version
   - Operating system

2. **Reproduction Steps**
   - Minimal reproducible code
   - Expected behavior
   - Actual behavior
   - Error messages

3. **Related Logs**
   - Complete error stack trace
   - Relevant configuration information

Example:
```python
# Minimal reproduction case
import genesis

model = genesis.nn.Linear(10, 5)
x = genesis.randn(3, 10)
y = model(x)  # Error occurs here

# Error message:
# RuntimeError: CUDA kernel launch failed
```

## 🎯 Key Contribution Areas

We particularly welcome contributions in the following areas:

### High Priority
- [ ] Performance optimization and benchmarking
- [ ] CUDA operator implementation
- [ ] Documentation and tutorial improvements
- [ ] Test coverage enhancement

### Medium Priority
- [ ] New neural network layers
- [ ] Data loader optimization
- [ ] Distributed training support
- [ ] Mixed precision training

### Low Priority
- [ ] Visualization tools
- [ ] Model deployment support
- [ ] Third-party framework integration

## 📞 Contact Us

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Technical discussions and Q&A
- **Email**: genesis-dev@example.com

## 🏆 Contributor Recognition

We value every contributor's effort:

- Contributors will be listed in the project README
- Major contributors will receive maintainer privileges
- Regular contributor newsletters

## 📄 License

By contributing code, you agree that your contributions will be licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

!!! info "Start Contributing"
    Ready to start contributing? Begin with [Development Environment Configuration](development.md)!

Thank you for your contribution to the Genesis project! 🎉