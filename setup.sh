#!/bin/bash

# Celebrity AI Art - Setup Script
echo "🎭 Celebrity AI Art Dataset Management System Setup"
echo "=================================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Check if we have Python 3.12+
if ! python3 -c "import sys; assert sys.version_info >= (3, 12)"; then
    echo "❌ Python 3.12+ is required!"
    echo "Please install Python 3.12 or higher and try again."
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "📦 Installing dependencies..."
pip install -e .

# Install development dependencies (optional)
read -p "Install development dependencies? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛠️ Installing development dependencies..."
    pip install -e ".[dev]"
fi

# Install training dependencies (optional)
read -p "Install training dependencies (for advanced features)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎯 Installing training dependencies..."
    pip install -e ".[training]"
fi

# Create example dataset directory
echo "📁 Creating example dataset directory..."
mkdir -p example_dataset

# Run basic usage example
echo "🚀 Running basic setup example..."
python examples/basic_usage.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run CLI help: celeb-ai --help"
echo "  3. Initialize dataset: celeb-ai init ./my_dataset"
echo "  4. Add celebrities: celeb-ai add-celebrity ./my_dataset --celebrity-id brad_pitt_001 --name 'Brad Pitt' --gender male --ethnicity caucasian"
echo "  5. Add raw images to: ./my_dataset/celebrities/brad_pitt_001/raw/high_quality/"
echo "  6. Process images: celeb-ai process ./my_dataset --celebrity-id brad_pitt_001"
echo ""
echo "📖 See README.md for detailed usage instructions"
echo ""
echo "🎉 Happy AI art generation!"
