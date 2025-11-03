#!/usr/bin/env python3
"""
Test script to verify unknown model device filtering
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.device_manager import DeviceManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Test unknown model device filtering"""
    print("🧪 Testing Unknown Model Device Filtering")
    print("=" * 50)
    
    try:
        # Test device discovery with filtering
        print("\n1. Testing device discovery...")
        device_manager = DeviceManager()
        
        # First, let's see what raw devices look like
        all_raw_devices = device_manager.adb.get_all_available_devices()
        print(f"   Raw devices found: {len(all_raw_devices)}")
        
        for i, device in enumerate(all_raw_devices, 1):
            print(f"   {i}. {device['id']} - Model: '{device.get('model', 'Not set')}'")
        
        # Now test filtered discovery
        print(f"\n2. Testing filtered discovery...")
        filtered_devices = device_manager.discover_devices()
        print(f"   Filtered devices: {len(filtered_devices)}")
        
        for i, device in enumerate(filtered_devices, 1):
            print(f"   {i}. {device['id']} - Model: '{device.get('model', 'Not set')}'")
        
        # Check if any unknown model devices were filtered out
        unknown_count = sum(1 for d in all_raw_devices if d.get('model') == 'Unknown' or not d.get('model'))
        filtered_count = len(all_raw_devices) - len(filtered_devices)
        
        print(f"\n3. Filtering results:")
        print(f"   Devices with unknown models: {unknown_count}")
        print(f"   Devices filtered out: {filtered_count}")
        
        if filtered_count == unknown_count:
            print(f"   ✅ Filtering working correctly!")
        elif filtered_count > 0:
            print(f"   ⚠️  Some devices were filtered (expected {unknown_count}, got {filtered_count})")
        else:
            print(f"   ℹ️  No devices were filtered (all have known models)")
        
        # Test game priority discovery
        print(f"\n4. Testing game priority discovery...")
        priority_devices = device_manager.discover_devices_with_game_priority()
        print(f"   Game priority devices: {len(priority_devices)}")
        
        for i, device in enumerate(priority_devices, 1):
            game_ready = "🎮" if device.get('game_ready') else "📱"
            print(f"   {i}. {game_ready} {device['id']} - Model: '{device.get('model', 'Not set')}'")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())