#!/usr/bin/env python3
"""
Test script to verify per-device bot control functionality
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.multi_device_manager import MultiDeviceManager
from src.core.device_manager import DeviceManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_device_control_commands():
    """Test device control commands"""
    print("🧪 Testing Per-Device Control Commands")
    print("=" * 50)
    
    try:
        # Initialize managers
        print("\n1. Initializing device managers...")
        device_manager = DeviceManager()
        multi_device_manager = MultiDeviceManager()
        
        # Discover and connect to devices
        print("\n2. Discovering devices...")
        devices = device_manager.discover_devices()
        print(f"   Found {len(devices)} devices with known models")
        
        if not devices:
            print("   ❌ No devices available for testing")
            return 1
        
        # Connect to first device for testing
        test_device = devices[0]
        device_id = test_device['id']
        print(f"\n3. Testing commands on device: {device_id}")
        
        success = multi_device_manager.connect_device(device_id, test_device)
        if not success:
            print(f"   ❌ Failed to connect to {device_id}")
            return 1
        
        print(f"   ✅ Connected to {device_id}")
        
        # Test screenshot command
        print(f"\n4. Testing screenshot command...")
        screenshot = multi_device_manager.execute_on_device(device_id, 'take_screenshot')
        if screenshot is not None:
            print(f"   ✅ Screenshot command works (shape: {screenshot.shape})")
        else:
            print(f"   ❌ Screenshot command failed")
        
        # Test tap command (safe coordinates - center of screen)
        print(f"\n5. Testing tap command...")
        tap_result = multi_device_manager.execute_on_device(device_id, 'tap', x=500, y=500)
        if tap_result:
            print(f"   ✅ Tap command works")
        else:
            print(f"   ❌ Tap command failed")
        
        # Test swipe command (safe swipe - small movement)
        print(f"\n6. Testing swipe command...")
        swipe_result = multi_device_manager.execute_on_device(device_id, 'swipe', x1=400, y1=400, x2=600, y2=400)
        if swipe_result:
            print(f"   ✅ Swipe command works")
        else:
            print(f"   ❌ Swipe command failed")
        
        # Test device status
        print(f"\n7. Testing device status...")
        device_status = multi_device_manager.get_device_status()
        print(f"   Connected devices: {device_status['connected_count']}")
        print(f"   Available slots: {device_status['available_slots']}")
        
        # Cleanup
        multi_device_manager.disconnect_device(device_id)
        print(f"\n✅ Device control testing completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return 1

def main():
    """Test per-device control functionality"""
    return test_device_control_commands()

if __name__ == "__main__":
    sys.exit(main())