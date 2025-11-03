#!/usr/bin/env python3
"""
Setup Script - Initialize project structure and configuration
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config_manager
from src.utils.logger import get_logger

def setup_project():
    """Setup the project structure and configuration"""
    print("🔧 Setting up Lineage 2M Bot Project")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    
    try:
        # Create necessary directories
        directories = [
            'screenshots',
            'logs',
            'assets/images',
            'assets/templates',
            'config'
        ]
        
        print("📁 Creating directories...")
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        
        # Initialize configuration
        print("\n⚙️ Initializing configuration...")
        config = config_manager.get_config()
        print(f"   ✅ Configuration loaded from: {config_manager.config_file}")
        
        # Setup logging
        print("\n📝 Setting up logging...")
        logger = get_logger("setup")
        logger.info("Project setup initiated")
        print("   ✅ Logging system initialized")
        
        # Create sample environment file
        env_file = project_root / '.env'
        if not env_file.exists():
            print("\n🔐 Creating sample environment file...")
            env_content = """# Lineage 2M Bot Environment Configuration
# Copy this file to .env and modify as needed

# ADB Configuration
LINEAGE2M_ADB_TIMEOUT=30

# Game Configuration
LINEAGE2M_GAME_PACKAGES=com.ncsoft.lineage2m,com.ncsoft.lineage2m.global

# Logging Configuration
LINEAGE2M_LOG_LEVEL=INFO

# Screenshot Configuration
LINEAGE2M_SAVE_SCREENSHOTS=true
LINEAGE2M_SCREENSHOT_DIR=screenshots
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("   ✅ .env.sample created")
        
        # Create gitignore
        gitignore_file = project_root / '.gitignore'
        if not gitignore_file.exists():
            print("\n📝 Creating .gitignore...")
            gitignore_content = """# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# Bot specific
logs/
screenshots/
temp_*.png
*.log

# Configuration
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_content)
            print("   ✅ .gitignore created")
        
        # Update requirements.txt
        print("\n📦 Updating requirements.txt...")
        requirements_file = project_root / 'requirements.txt'
        requirements_content = """# Core dependencies
adb-shell==0.4.4
opencv-python==4.8.1.78
numpy==1.24.3
pillow==10.0.1
psutil==5.9.5

# Configuration and utilities
pyyaml>=6.0
python-dotenv>=1.0.0

# Optional dependencies for advanced features
# pytesseract>=0.3.10  # For OCR functionality
# mss>=7.0.1          # For faster screenshots
"""
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        print("   ✅ requirements.txt updated")
        
        print("\n🎉 Project setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Review config/bot_config.yaml")
        print("   2. Copy .env.sample to .env and customize")
        print("   3. Run: python scripts/test_connection.py")
        print("   4. Run: python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_project()
    exit(0 if success else 1)