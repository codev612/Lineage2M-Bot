#!/usr/bin/env python3
"""
Test device manager with fresh instance after config update
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.device_manager import DeviceManager
from src.utils.config import config_manager

def main():
    print("🧪 Testing device manager with fresh instance...")
    
    # Force reload config first
    config_manager.reload_config()
    
    # Get packages from reloaded config
    game_config = config_manager.get_game_config()
    packages = game_config.packages
    print(f"📦 Config packages: {packages}")
    
    # Create fresh device manager
    device_manager = DeviceManager()
    
    # Check if device manager got the correct config
    dm_packages = device_manager.config.game.packages
    print(f"📦 Device manager packages: {dm_packages}")
    print(f"✅ Packages match: {packages == dm_packages}")
    
    # Test device discovery
    print(f"\n🔍 Running device discovery...")
    devices = device_manager.discover_devices()
    
    # Find our test device
    test_device = None
    for device in devices:
        if device['id'] == '127.0.0.1:5555':
            test_device = device
            break
    
    if test_device:
        print(f"\n📱 Found test device: {test_device['id']}")
        print(f"   Model: {test_device.get('model', 'Unknown')}")
        print(f"   Game status: {test_device.get('game_status', 'Not available')}")
    else:
        print(f"\n❌ Test device 127.0.0.1:5555 not found")

if __name__ == "__main__":
    main()