#!/usr/bin/env python3
"""
Debug device discovery and connection status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.adb_manager import ADBManager
from src.utils.config import config_manager

def main():
    print("🧪 Debug device discovery and connection status...")
    
    # Force reload config
    config_manager.reload_config()
    game_packages = config_manager.get_game_config().packages
    print(f"📦 Game packages: {game_packages}")
    
    # Create ADB manager
    adb = ADBManager()
    
    # Get all available devices
    print(f"\n🔍 Getting all available devices...")
    devices = adb.get_all_available_devices()
    
    # Find our test device
    test_device = None
    for device in devices:
        if device['id'] == '127.0.0.1:5555':
            test_device = device
            break
    
    if test_device:
        print(f"📱 Found test device: {test_device}")
        
        device_id = test_device['id']
        status = test_device.get('status', 'unknown')
        
        print(f"\n🔍 Device status: {status}")
        
        # Try game status check without explicit connection
        print(f"🎮 Testing game status without explicit connection...")
        game_status_1 = adb.check_game_status(device_id, game_packages)
        print(f"   Result: {game_status_1}")
        
        # Try connecting explicitly first
        print(f"\n🔗 Connecting to device explicitly...")
        connect_success = adb.connect_to_device(device_id)
        print(f"   Connection success: {connect_success}")
        
        # Try game status check after explicit connection
        print(f"🎮 Testing game status after explicit connection...")
        game_status_2 = adb.check_game_status(device_id, game_packages)
        print(f"   Result: {game_status_2}")
        
    else:
        print(f"❌ Test device not found in available devices")

if __name__ == "__main__":
    main()