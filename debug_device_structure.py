#!/usr/bin/env python3
"""
Debug device object structure after discovery
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.device_manager import DeviceManager
from src.utils.config import config_manager
import json

def main():
    print("🧪 Debug device object structure...")
    
    # Force reload config
    config_manager.reload_config()
    
    # Create device manager and discover devices
    device_manager = DeviceManager()
    devices = device_manager.discover_devices_with_game_priority()
    
    # Find our test device
    test_device = None
    for device in devices:
        if device['id'] == '127.0.0.1:5555':
            test_device = device
            break
    
    if test_device:
        print(f"📱 Found test device structure:")
        print(json.dumps(test_device, indent=2, default=str))
    else:
        print(f"❌ Test device not found")

if __name__ == "__main__":
    main()