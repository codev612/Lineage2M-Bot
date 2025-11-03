#!/usr/bin/env python3
"""
Direct test of ADB game status check with updated configuration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.adb_manager import ADBManager
from src.utils.config import config_manager

def main():
    print("🧪 Testing ADB game status with updated config...")
    
    # Show loaded packages
    game_config = config_manager.get_game_config()
    print(f"📦 Loaded packages: {game_config.packages}")
    
    # Test ADB manager
    adb = ADBManager()
    device_id = "127.0.0.1:5555"
    
    print(f"\n🔗 Testing device: {device_id}")
    
    # Connect first
    success = adb.connect_to_device(device_id)
    print(f"   📱 Connection: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        # Test game status
        status = adb.check_game_status(device_id)
        print(f"   🎮 Game status: {status}")
        
        # Test individual methods
        installed, installed_packages = adb._is_game_installed(device_id)
        print(f"   📥 Installed: {installed} (packages: {installed_packages})")
        
        running, running_packages = adb._is_game_running(device_id)
        print(f"   🏃 Running: {running} (packages: {running_packages})")

if __name__ == "__main__":
    main()