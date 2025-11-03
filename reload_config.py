#!/usr/bin/env python3
"""
Script to force reload the global configuration manager
This fixes the singleton caching issue when config files are updated
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config_manager
from src.core.adb_manager import ADBManager

def main():
    print("🔄 Force reloading configuration...")
    
    # Force reload the global config manager
    config_manager.reload_config()
    
    # Test the reloaded configuration
    game_config = config_manager.get_game_config()
    packages = game_config.packages
    
    print(f"📦 Packages after reload: {packages}")
    print(f"✅ Has lineage2mnu: {'com.ncsoft.lineage2mnu' in packages}")
    
    # Test with ADB manager
    print("\n🧪 Testing with ADB manager...")
    adb = ADBManager()
    adb.connect_to_device("127.0.0.1:5555")
    
    installed, packages_found = adb.is_game_installed("127.0.0.1:5555")
    running, running_packages = adb.is_game_running("127.0.0.1:5555")
    
    print(f"   🔗 Device: 127.0.0.1:5555")
    print(f"   📱 Installed: {installed}")
    print(f"   📦 Packages found: {packages_found}")
    print(f"   🏃 Running: {running}")
    print(f"   ⚡ Running packages: {running_packages}")
    
    print("\n✅ Configuration reloaded successfully!")

if __name__ == "__main__":
    main()