#!/usr/bin/env python3
"""
Test script for multi-device functionality
Demonstrates parallel device management and control
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.multi_device_manager import MultiDeviceManager
from src.utils.config import config_manager
import time

def test_multi_device_manager():
    print("🧪 Testing Multi-Device Manager...")
    
    # Initialize multi-device manager
    mdm = MultiDeviceManager()
    
    try:
        print("\n1️⃣ Discovering devices...")
        devices = mdm.discover_devices()
        print(f"   Found {len(devices)} device(s)")
        
        # Show available devices
        if devices:
            print("\n📱 Available devices:")
            for i, device in enumerate(devices, 1):
                game_status = device.get('game_status', {})
                status_text = "🎮 Game running" if game_status.get('running') else "📱 Game installed" if game_status.get('installed') else "No game"
                print(f"   {i}. {device['id']} - {device.get('model', 'Unknown')} ({status_text})")
            
            # Test connecting to multiple devices
            print(f"\n2️⃣ Testing multi-device connections...")
            connected_count = 0
            
            # Try to connect to first few devices
            for device in devices[:3]:  # Limit to first 3 devices
                device_id = device['id']
                print(f"   Connecting to {device_id}...")
                
                if mdm.connect_device(device_id, device):
                    connected_count += 1
                    print(f"   ✅ Connected to {device_id}")
                    
                    # Select this device
                    mdm.select_device(device_id)
                else:
                    print(f"   ❌ Failed to connect to {device_id}")
            
            print(f"\n📊 Connection Results:")
            print(f"   🟢 Connected devices: {connected_count}")
            
            # Test parallel operations
            if connected_count > 0:
                print(f"\n3️⃣ Testing parallel operations...")
                
                # Get connected devices info
                connected_devices = mdm.get_connected_devices()
                print(f"   Connected devices: {list(connected_devices.keys())}")
                
                # Test simple command execution
                print(f"   Testing parallel 'echo test' command...")
                results = mdm.execute_on_selected(['shell', 'echo', 'test'])
                
                for device_id, (success, output) in results.items():
                    status = "✅" if success else "❌"
                    print(f"   {status} {device_id}: {output.strip()}")
                
                # Test game status check
                print(f"   Testing parallel game status check...")
                game_statuses = mdm.get_game_status_all()
                
                for device_id, status in game_statuses.items():
                    if 'error' in status:
                        print(f"   ❌ {device_id}: {status['error']}")
                    else:
                        running = "🎮 Running" if status.get('game_running') else "⏹️ Not running"
                        installed = "📱 Installed" if status.get('game_installed') else "❌ Not installed"
                        print(f"   📊 {device_id}: {installed}, {running}")
                
                print(f"\n4️⃣ Testing device selection management...")
                selected_devices = mdm.get_selected_devices()
                print(f"   Currently selected: {selected_devices}")
                
                # Test select all
                mdm.select_all_connected()
                selected_devices = mdm.get_selected_devices()
                print(f"   After select all: {selected_devices}")
                
                # Test deselect all
                mdm.deselect_all()
                selected_devices = mdm.get_selected_devices()
                print(f"   After deselect all: {selected_devices}")
                
        else:
            print("   ⚠️ No devices found for testing")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n🧹 Cleaning up...")
        mdm.cleanup()
        print("✅ Multi-device manager test completed!")

if __name__ == "__main__":
    test_multi_device_manager()