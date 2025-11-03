#!/usr/bin/env python3
"""
Debug script to check device control widget flow
"""

import sys
from pathlib import Path

# Add the src directory to the Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.device_manager import DeviceManager
from src.core.multi_device_manager import MultiDeviceManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def debug_device_control_flow():
    """Debug the device control flow"""
    print("🐛 Debugging Device Control Flow")
    print("=" * 50)
    
    try:
        # Step 1: Device Discovery
        print("\n1. Testing device discovery...")
        device_manager = DeviceManager()
        devices = device_manager.discover_devices_with_game_priority()
        
        print(f"   Discovered {len(devices)} devices:")
        for i, device in enumerate(devices, 1):
            game_status = device.get('game_status', {})
            game_info = "🎮 running" if game_status.get('running') else "📱 installed" if game_status.get('installed') else "❌ not installed"
            print(f"   {i}. {device['id']} ({device.get('model', 'Unknown')}) - {game_info}")
        
        # Step 2: Multi-device manager setup
        print(f"\n2. Testing multi-device manager...")
        multi_device_manager = MultiDeviceManager()
        device_status = multi_device_manager.get_device_status()
        
        print(f"   Connected devices: {device_status['connected_count']}")
        print(f"   Max devices: {device_status['max_devices']}")
        print(f"   Available slots: {device_status['available_slots']}")
        
        # Step 3: Test device connection
        if devices:
            test_device = devices[0]
            device_id = test_device['id']
            print(f"\n3. Testing device connection to {device_id}...")
            
            success = multi_device_manager.connect_device(device_id, test_device)
            if success:
                print(f"   ✅ Successfully connected to {device_id}")
                
                # Test getting connected devices
                connected = multi_device_manager.get_connected_devices()
                print(f"   Connected devices: {list(connected.keys())}")
                
                # Test device control widget requirements
                print(f"\n4. Device control widget requirements:")
                print(f"   Device needs to be:")
                print(f"   - Selected in GUI device tree (simulated: ✅)")
                print(f"   - Connected via multi-device manager: ✅")
                print(f"   → Widget should appear for this device")
                
                # Cleanup
                multi_device_manager.disconnect_device(device_id)
                print(f"\n   Disconnected from {device_id}")
            else:
                print(f"   ❌ Failed to connect to {device_id}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        print(f"\n❌ Debug failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(debug_device_control_flow())