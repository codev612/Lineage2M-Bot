#!/usr/bin/env python3
"""
Test script to verify device limit configuration
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config_manager
from src.core.multi_device_manager import MultiDeviceManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Test device limit configuration"""
    print("🧪 Testing Device Limit Configuration")
    print("=" * 50)
    
    try:
        # Test configuration loading
        print("\n1. Testing configuration loading...")
        config = config_manager.get_config()
        max_devices = config.max_devices
        print(f"   ✅ Device limit configured: {max_devices}")
        
        # Test MultiDeviceManager
        print("\n2. Testing MultiDeviceManager...")
        manager = MultiDeviceManager()
        device_status = manager.get_device_status()
        
        print(f"   ✅ Current connected devices: {device_status['connected_count']}")
        print(f"   ✅ Maximum allowed devices: {device_status['max_devices']}")
        print(f"   ✅ Available slots: {device_status['available_slots']}")
        
        # Verify the configuration values match
        if device_status['max_devices'] == max_devices:
            print(f"   ✅ Configuration consistency verified!")
        else:
            print(f"   ❌ Configuration mismatch!")
            
        # Test device limit enforcement (simulated)
        print("\n3. Testing device limit logic...")
        if device_status['available_slots'] == 0:
            print(f"   ⚠️  Device limit reached - new connections will be blocked")
        else:
            print(f"   ✅ {device_status['available_slots']} slots available for new connections")
            
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())