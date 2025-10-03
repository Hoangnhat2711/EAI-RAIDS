#!/bin/bash

# EAI-RAIDS Installation Script
# Version: 3.1.0
# Last Updated: October 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║              🚀 EAI-RAIDS Installation Script 🚀                 ║"
echo "║                                                                   ║"
echo "║        Enterprise Responsible AI Detection & Security System     ║"
echo "║                        Version 3.1.0                              ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo -e "${BLUE}📋 Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Error: Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    echo -e "${YELLOW}💡 Please install Python 3.8 or higher${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"
fi

# Installation mode selection
echo ""
echo -e "${BLUE}📦 Select installation mode:${NC}"
echo "  1) Basic     - Core features only (fastest)"
echo "  2) Standard  - Core + SOTA features (recommended)"
echo "  3) Full      - All features including optional dependencies"
echo "  4) Custom    - Select components manually"
echo ""
read -p "Enter choice [1-4] (default: 2): " INSTALL_MODE
INSTALL_MODE=${INSTALL_MODE:-2}

# Function to install package with progress
install_package() {
    local package=$1
    local description=$2
    echo -e "${BLUE}  📦 Installing $description...${NC}"
    pip install $package --quiet || {
        echo -e "${RED}  ❌ Failed to install $package${NC}"
        return 1
    }
    echo -e "${GREEN}  ✅ $description installed${NC}"
}

# Basic installation
install_basic() {
    echo ""
    echo -e "${BLUE}🔧 Installing BASIC packages...${NC}"
    echo ""
    
    install_package "numpy>=1.21.0" "NumPy"
    install_package "pandas>=1.3.0" "Pandas"
    install_package "scikit-learn>=1.0.0" "Scikit-learn"
    install_package "scipy>=1.7.0" "SciPy"
    install_package "pyyaml>=5.4.0" "PyYAML"
    install_package "joblib>=1.1.0" "Joblib"
    
    echo ""
    echo -e "${GREEN}✅ Basic installation complete!${NC}"
}

# Standard installation
install_standard() {
    install_basic
    
    echo ""
    echo -e "${BLUE}🌟 Installing SOTA features...${NC}"
    echo ""
    
    install_package "matplotlib>=3.4.0" "Matplotlib"
    install_package "seaborn>=0.11.0" "Seaborn"
    install_package "shap>=0.41.0" "SHAP"
    install_package "lime>=0.2.0" "LIME"
    install_package "fairlearn>=0.7.0" "Fairlearn"
    install_package "imbalanced-learn>=0.9.0" "Imbalanced-learn"
    
    echo ""
    echo -e "${GREEN}✅ Standard installation complete!${NC}"
}

# Full installation
install_full() {
    install_standard
    
    echo ""
    echo -e "${BLUE}🚀 Installing FULL features (this may take a while)...${NC}"
    echo ""
    
    # Deep Learning
    echo -e "${BLUE}  🤖 Deep Learning frameworks...${NC}"
    install_package "torch>=1.10.0" "PyTorch"
    install_package "tensorflow>=2.8.0" "TensorFlow"
    
    # Privacy
    echo ""
    echo -e "${BLUE}  🔒 Privacy libraries...${NC}"
    install_package "opacus>=1.3.0" "Opacus (DP-SGD for PyTorch)"
    install_package "tensorflow-privacy>=0.8.0" "TensorFlow Privacy"
    install_package "cryptography>=36.0.0" "Cryptography"
    
    # Causal Inference
    echo ""
    echo -e "${BLUE}  🧠 Causal inference libraries...${NC}"
    install_package "dowhy>=0.9.0" "DoWhy"
    install_package "causalml>=0.14.0" "CausalML"
    install_package "causal-learn>=0.1.3" "Causal-learn"
    install_package "lingam>=1.8.0" "LiNGAM"
    
    # MLOps
    echo ""
    echo -e "${BLUE}  🔧 MLOps tools...${NC}"
    install_package "mlflow>=2.0.0" "MLflow"
    install_package "dvc>=3.0.0" "DVC"
    
    # Testing
    echo ""
    echo -e "${BLUE}  🧪 Testing tools...${NC}"
    install_package "pytest>=7.0.0" "Pytest"
    install_package "pytest-cov>=4.0.0" "Pytest Coverage"
    
    echo ""
    echo -e "${GREEN}✅ Full installation complete!${NC}"
}

# Custom installation
install_custom() {
    echo ""
    echo -e "${BLUE}📦 Custom installation${NC}"
    echo ""
    
    # Always install core
    install_basic
    
    # Ask for each category
    echo ""
    read -p "Install Deep Learning (PyTorch, TensorFlow)? [y/N]: " install_dl
    if [[ $install_dl =~ ^[Yy]$ ]]; then
        install_package "torch>=1.10.0" "PyTorch"
        install_package "tensorflow>=2.8.0" "TensorFlow"
    fi
    
    echo ""
    read -p "Install Explainability (SHAP, LIME)? [y/N]: " install_xai
    if [[ $install_xai =~ ^[Yy]$ ]]; then
        install_package "shap>=0.41.0" "SHAP"
        install_package "lime>=0.2.0" "LIME"
    fi
    
    echo ""
    read -p "Install Privacy (Opacus, TF Privacy)? [y/N]: " install_privacy
    if [[ $install_privacy =~ ^[Yy]$ ]]; then
        install_package "opacus>=1.3.0" "Opacus"
        install_package "tensorflow-privacy>=0.8.0" "TensorFlow Privacy"
    fi
    
    echo ""
    read -p "Install Causal Inference (DoWhy, CausalML)? [y/N]: " install_causal
    if [[ $install_causal =~ ^[Yy]$ ]]; then
        install_package "dowhy>=0.9.0" "DoWhy"
        install_package "causalml>=0.14.0" "CausalML"
        install_package "causal-learn>=0.1.3" "Causal-learn"
        install_package "lingam>=1.8.0" "LiNGAM"
    fi
    
    echo ""
    read -p "Install MLOps (MLflow, DVC)? [y/N]: " install_mlops
    if [[ $install_mlops =~ ^[Yy]$ ]]; then
        install_package "mlflow>=2.0.0" "MLflow"
        install_package "dvc>=3.0.0" "DVC"
    fi
    
    echo ""
    echo -e "${GREEN}✅ Custom installation complete!${NC}"
}

# Run installation based on mode
case $INSTALL_MODE in
    1)
        install_basic
        ;;
    2)
        install_standard
        ;;
    3)
        install_full
        ;;
    4)
        install_custom
        ;;
    *)
        echo -e "${RED}❌ Invalid choice. Defaulting to Standard installation.${NC}"
        install_standard
        ;;
esac

# Verify installation
echo ""
echo -e "${BLUE}🔍 Verifying installation...${NC}"
echo ""

python3 << EOF
import sys
success_count = 0
fail_count = 0

modules = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'Scikit-learn'),
    ('scipy', 'SciPy'),
]

print("Core modules:")
for module, name in modules:
    try:
        __import__(module)
        print(f"  ✅ {name}")
        success_count += 1
    except ImportError:
        print(f"  ❌ {name}")
        fail_count += 1

print(f"\n📊 Results: {success_count} ✅, {fail_count} ❌")

if fail_count == 0:
    print("\n🎉 All core modules installed successfully!")
else:
    print(f"\n⚠️  {fail_count} module(s) failed to install")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Installation verification passed!${NC}"
else
    echo ""
    echo -e "${RED}❌ Installation verification failed!${NC}"
    echo -e "${YELLOW}💡 Try running: pip install -r requirements.txt${NC}"
    exit 1
fi

# Create directories
echo ""
echo -e "${BLUE}📁 Creating project directories...${NC}"
mkdir -p audit_logs
mkdir -p experiments
mkdir -p mlruns
echo -e "${GREEN}✅ Directories created${NC}"

# Final summary
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║                  🎉 Installation Complete! 🎉                    ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ EAI-RAIDS is ready to use!${NC}"
echo ""
echo -e "${BLUE}📚 Next steps:${NC}"
echo "  1. Run demo:           python3 examples/demo.py"
echo "  2. Run advanced demo:  python3 examples/advanced_demo.py"
echo "  3. Run tests:          pytest tests/"
echo "  4. Read docs:          cat README.md"
echo ""
echo -e "${BLUE}🔗 Resources:${NC}"
echo "  • Repository:  https://github.com/Hoangnhat2711/EAI-RAIDS"
echo "  • Docs:        All *.md files"
echo "  • Examples:    examples/ directory"
echo ""
echo -e "${YELLOW}💡 Pro tip: Run 'python3 basic_demo.py' for a quick system check!${NC}"
echo ""

