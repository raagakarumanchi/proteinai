# Contributing to FoldAI

Thank you for your interest in contributing to FoldAI! We welcome contributions from the community and are excited to work with you.

## How to Contribute

### **Reporting Bugs**
- Use the GitHub issue tracker
- Include detailed steps to reproduce
- Provide system information (OS, Python version, etc.)
- Include error messages and logs

### **Suggesting Features**
- Open a feature request issue
- Describe the use case and benefits
- Consider implementation complexity
- Check if similar features exist

### **Code Contributions**
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests if applicable
- Submit a pull request

## Development Setup

### **Prerequisites**
- Python 3.8 or higher
- Git
- Basic understanding of machine learning

### **Setup Steps**
```bash
# 1. Fork and clone
git clone https://github.com/yourusername/foldai.git
cd foldai

# 2. Create virtual environment
python -m venv foldai_env
source foldai_env/bin/activate  # On Windows: foldai_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test installation
python src/main.py
```

## Code Style

### **Python Standards**
- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for functions
- Keep functions focused and small

### **Example Code Style**
```python
def predict_protein_function(sequence: str) -> Dict[str, float]:
    """
    Predict protein function from amino acid sequence.
    
    Args:
        sequence: Protein sequence as string
        
    Returns:
        Dictionary with function predictions and confidence scores
    """
    # Implementation here
    pass
```

## Testing

### **Adding Tests**
- Create tests in `tests/` directory
- Use pytest framework
- Test both success and failure cases
- Include edge cases

### **Running Tests**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## Documentation

### **Code Documentation**
- Add docstrings to all functions
- Include parameter descriptions
- Provide usage examples
- Update README for new features

### **API Documentation**
- Document new API endpoints
- Include request/response examples
- Update OpenAPI specs if applicable

## Areas for Contribution

### **High Priority**
- **New AI Models**: Implement cutting-edge architectures
- **Visualizations**: Create better protein visualizations
- **Testing**: Add comprehensive test suite
- **Documentation**: Improve API documentation

### **Medium Priority**
- **Web Interface**: Improve Flask web app
- **Mobile App**: Build React Native version
- **Performance**: Optimize model inference
- **DevOps**: CI/CD pipeline setup

### **Low Priority**
- **UI/UX**: Improve user interface
- **Internationalization**: Multi-language support
- **Analytics**: Usage tracking and metrics
- **Security**: Security audit and improvements

## Community Guidelines

### **Be Respectful**
- Use inclusive language
- Be patient with newcomers
- Provide constructive feedback
- Respect different opinions

### **Communication**
- Use clear, concise language
- Provide context for discussions
- Ask questions when unsure
- Share knowledge and resources

## Recognition

### **Contributor Benefits**
- Listed in CONTRIBUTORS.md
- Featured in release notes
- Mentioned in social media
- Eligible for rewards

### **Contributor Levels**
- **Bronze**: 1-5 contributions
- **Silver**: 6-15 contributions  
- **Gold**: 16+ contributions
- **Platinum**: Major feature contributions

## Getting Help

### **Resources**
- Documentation: README.md
- Discussions: GitHub Discussions
- Issues: GitHub Issues
- Email: contact@foldai.com

### **Community**
- Star the repository
- Fork for your projects
- Share on social media
- Connect with other contributors

Thank you for contributing to FoldAI! Together, we're advancing protein science with AI.
