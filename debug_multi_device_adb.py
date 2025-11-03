#!/usr/bin/env python3
"""
Debug ADB command execution in multi-device setup
"""

import sys
from pathlib import Path

# Add the src directory to the Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.multi_device_manager import MultiDeviceManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def debug_multi_device_adb():
    """Debug ADB command execution through multi-device manager"""
    print("🔧 Debugging Multi-Device ADB Command Execution")
    print("=" * 60)
    
    try:
        # Initialize multi-device manager
        print("1. Initializing MultiDeviceManager...")
        mdm = MultiDeviceManager()
        
        # Get connected devices
        print("2. Getting connected devices...")
        connected_devices = mdm.get_connected_devices()
        print(f"   Connected devices: {list(connected_devices.keys())}")
        
        if not connected_devices:
            print("   ❌ No devices connected through MultiDeviceManager")
            print("   Trying to discover and connect...")
            
            devices = mdm.discover_devices()
            print(f"   Discovered {len(devices)} devices")
            
            for device in devices:
                device_id = device['id']
                print(f"   Attempting to connect to {device_id}...")
                success = mdm.connect_device(device_id, device)
                if success:
                    print(f"   ✅ Connected to {device_id}")
                    break
                else:
                    print(f"   ❌ Failed to connect to {device_id}")
        
        # Test command execution
        connected_devices = mdm.get_connected_devices()
        if connected_devices:
            test_device_id = list(connected_devices.keys())[0]
            print(f"\n3. Testing command execution on {test_device_id}...")
            
            # Test tap command
            print("   Testing tap command...")
            try:
                result = mdm.execute_on_device(test_device_id, 'tap', x=540, y=960)
                print(f"   Tap result: {result}")
                
                if result is True:
                    print("   ✅ Tap command executed successfully")
                elif result is False:
                    print("   ❌ Tap command failed")
                else:
                    print(f"   ⚠️  Unexpected tap result: {result}")
                    
            except Exception as e:
                print(f"   ❌ Tap command error: {e}")
            
            # Test swipe command
            print("   Testing swipe command...")
            try:
                result = mdm.execute_on_device(test_device_id, 'swipe', x1=300, y1=960, x2=780, y2=960)
                print(f"   Swipe result: {result}")
                
                if result is True:
                    print("   ✅ Swipe command executed successfully")
                elif result is False:
                    print("   ❌ Swipe command failed")
                else:
                    print(f"   ⚠️  Unexpected swipe result: {result}")
                    
            except Exception as e:
                print(f"   ❌ Swipe command error: {e}")
        else:
            print("   ❌ No devices available for testing")
    
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        logger.error(f"Debug error: {e}")

if __name__ == "__main__":
    debug_multi_device_adb()