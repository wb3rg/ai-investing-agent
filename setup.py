#!/usr/bin/env python3
"""
Setup script for AI Investment Agent Framework
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🤖 AI Investment Agent Framework Setup")
    print("Using Agno + DeepSeek R1 for Intelligent Investment Analysis")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        sys.exit(1)

def setup_environment():
    """Setup environment configuration"""
    print("\n🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("Creating .env file for API keys...")
        with open(".env", "w") as f:
            f.write("# AI Investment Agent Framework Environment Variables\n")
            f.write("# Get your OpenRouter API key from: https://openrouter.ai/\n")
            f.write("OPENROUTER_API_KEY=your_openrouter_api_key_here\n\n")
            f.write("# Optional: Financial Datasets API key from: https://financialdatasets.ai/\n")
            f.write("FINANCIAL_DATASETS_API_KEY=your_financial_datasets_api_key_here\n\n")
            f.write("# Optional: Other API keys\n")
            f.write("# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key\n")
            f.write("# OPENBB_API_KEY=your_openbb_key\n")
        
        print("✅ Created .env file")
        print("⚠️  Please edit .env file and add your API keys")
    else:
        print("✅ .env file already exists")

def verify_installation():
    """Verify that the installation was successful"""
    print("\n🔍 Verifying installation...")
    
    try:
        import agno
        print("✅ Agno framework imported successfully")
    except ImportError:
        print("❌ Failed to import Agno framework")
        return False
    
    try:
        import yfinance
        print("✅ YFinance imported successfully")
    except ImportError:
        print("❌ Failed to import YFinance")
        return False
    
    try:
        import pandas
        import numpy
        import scipy
        print("✅ Data analysis libraries imported successfully")
    except ImportError:
        print("❌ Failed to import data analysis libraries")
        return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup complete! Next steps:")
    print("=" * 60)
    print("1. Edit the .env file and add your API keys:")
    print("   - Get OpenRouter API key from: https://openrouter.ai/")
    print("   - (Optional) Get Financial Datasets API key from: https://financialdatasets.ai/")
    print()
    print("2. Set environment variables:")
    print("   export $(cat .env | xargs)")
    print("   # Or source the .env file in your shell")
    print()
    print("3. Run the example:")
    print("   python investment_agent_example.py")
    print()
    print("4. Check the documentation:")
    print("   - Read ai_investing_agent_framework_guide.md")
    print("   - Review the example code in investment_agent_example.py")
    print()
    print("💡 Tips:")
    print("- Start with the free DeepSeek R1 model (deepseek/deepseek-r1-0528:free)")
    print("- YFinance provides free financial data")
    print("- Consider upgrading to paid APIs for production use")
    print("=" * 60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Setup environment
    setup_environment()
    
    # Verify installation
    if verify_installation():
        print("✅ Installation verified successfully")
    else:
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
