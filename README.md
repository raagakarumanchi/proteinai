# FoldAI - Protein Function Predictor

A Python-based platform for protein function prediction using machine learning. This project implements various approaches to predict protein functions from amino acid sequences.

## Features

- **Structure Prediction**: Secondary structure prediction using Chou-Fasman algorithm
- **Function Prediction**: Basic protein function classification
- **Deep Learning Models**: Transformer and CNN-based approaches
- **Data Integration**: UniProt API integration for protein data
- **Visualizations**: Interactive plots and 3D structure visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/raagakarumanchi/proteinai.git
cd proteinai

# Install dependencies
pip install -r requirements.txt

# Run the main application
python src/main.py
```

## Running Locally

### **Quick Start (30 seconds)**
```bash
# Clone the repository
git clone https://github.com/yourusername/foldai.git
cd foldai

# Install dependencies
pip install -r requirements.txt

# Run the advanced AI predictor
python src/main_advanced.py
```

### **Full Setup (5 minutes)**
```bash
# 1. Clone and navigate
git clone https://github.com/yourusername/foldai.git
cd foldai

# 2. Create virtual environment (recommended)
python -m venv foldai_env
source foldai_env/bin/activate  # On Windows: foldai_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the advanced model (optional)
python train_advanced_model.py

# 5. Run predictions
python src/main_advanced.py
```

### **Different Versions Available**
```bash
# Advanced multi-modal AI (recommended)
python src/main_advanced.py

# Basic function prediction
python src/main_function.py

# Simple structure prediction
python src/main.py

# Viral demo
python examples/viral_demo.py
```

### **System Requirements**
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (CUDA-compatible)
- **Storage**: 2GB free space

### **Troubleshooting**
```bash
# If you get import errors
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# If you get CUDA errors (GPU issues)
# The system will automatically fall back to CPU

# If you get memory errors
# Reduce batch size in the code or use smaller sequences
```

## Super Simple Structure

```
foldai/
├── src/
│   ├── main.py              # One-click viral demo
│   ├── analysis/            # AI prediction engine
│   ├── data_fetchers/       # Protein data APIs
│   └── visualization/       # Beautiful 3D plots
├── examples/                # Try it yourself
└── data/                    # Your results
```

## Available Models

1. **Structure Predictor** - Secondary structure prediction
2. **Function Predictor** - Basic protein function classification
3. **Deep Learning Models** - Transformer and CNN architectures
4. **Visualization Tools** - Interactive protein analysis plots

## Perfect For

- **Students** learning protein science
- **Researchers** needing quick predictions  
- **Developers** building bio apps
- **Anyone** curious about proteins!

## Project Structure

```
foldai/
├── src/
│   ├── main.py                          # Simple structure prediction
│   ├── main_function.py                 # Basic function prediction
│   ├── main_advanced.py                 # Advanced multi-modal AI
│   ├── analysis/
│   │   ├── structure_predictor.py       # Structure prediction algorithms
│   │   ├── protein_function_predictor.py # Basic function prediction
│   │   ├── advanced_protein_predictor.py # Advanced multi-modal AI
│   │   └── deep_learning_predictor.py   # Deep learning models
│   ├── data_fetchers/
│   │   └── uniprot_client.py           # Protein data APIs
│   ├── visualization/
│   │   └── sequence_plots.py           # Interactive visualizations
│   └── utils/
│       └── metrics.py                  # Evaluation metrics
├── examples/
│   ├── protein_structure_demo.py       # Structure analysis demo
│   └── viral_demo.py                   # Viral demo
├── models/                             # Trained model weights
├── data/                               # Generated results and cache
├── train_advanced_model.py             # Advanced model training
├── train_models.py                     # Basic model training
├── web_app.py                          # Web interface
├── run.py                              # Quick start script
├── requirements.txt                    # Dependencies
└── README.md                           # Documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **What this means:**
- **Free to use** for personal and commercial projects
- **Modify and distribute** as needed
- **Private use** allowed
- **Attribution** required (credit FoldAI)

### **Commercial Use:**
- **Startups**: Use freely in your products
- **Companies**: Integrate into commercial software
- **Biotech**: Deploy in research and development
- **Pharma**: Use for drug discovery pipelines

## Contributing

We welcome contributions! Here's how to get involved:

### **Ways to Contribute:**
- **Report bugs** - Open an issue with detailed description
- **Suggest features** - Propose new AI models or capabilities
- **Fix issues** - Submit pull requests for bug fixes
- **Improve docs** - Enhance documentation and examples
- **Add tests** - Improve test coverage
- **Share results** - Post your protein predictions

### **Development Setup:**
```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/yourusername/foldai.git
cd foldai

# 3. Create feature branch
git checkout -b feature/amazing-feature

# 4. Make changes and test
python src/main_advanced.py

# 5. Commit changes
git commit -m "Add amazing feature"

# 6. Push to branch
git push origin feature/amazing-feature

# 7. Open Pull Request
```

### **Contribution Guidelines:**
- **Code Style**: Follow PEP 8 Python standards
- **Testing**: Add tests for new features
- **Documentation**: Update README for new features
- **Focus**: Keep changes focused and well-documented
- **Be Respectful**: Follow our code of conduct

### **Areas Needing Help:**
- **New AI Models**: Implement cutting-edge architectures
- **Visualizations**: Create better protein visualizations
- **Web Interface**: Improve the Flask web app
- **Mobile App**: Build React Native mobile version
- **Testing**: Add comprehensive test suite
- **Documentation**: Improve API documentation

### **Recognition:**
Contributors will be:
- **Listed** in CONTRIBUTORS.md
- **Featured** in release notes
- **Mentioned** in social media posts
- **Eligible** for contributor rewards

## Contact

For questions or issues, please open an issue on GitHub.
