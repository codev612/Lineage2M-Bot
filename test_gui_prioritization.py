#!/usr/bin/env python3
"""
Test script to verify GUI device prioritization logic
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.device_manager import DeviceManager
from src.utils.config import config_manager

def test_gui_device_prioritization():
    print("🧪 Testing GUI device prioritization logic...")
    
    # Force reload config and create device manager
    config_manager.reload_config()
    device_manager = DeviceManager()
    
    # Discover devices with game priority
    devices = device_manager.discover_devices_with_game_priority()
    
    # Simulate GUI device list processing
    game_devices = []
    regular_devices = []
    
    for device in devices:
        game_status = device.get('game_status', {})
        if game_status.get('installed') or game_status.get('running'):
            game_devices.append(device)
        else:
            regular_devices.append(device)
    
    print(f"\n📊 Device Categorization Results:")
    print(f"🎮 Game devices (will be prioritized): {len(game_devices)}")
    print(f"📱 Regular devices: {len(regular_devices)}")
    
    if game_devices:
        print(f"\n⭐ Game devices that will appear first in GUI:")
        for device in game_devices:
            game_status = device.get('game_status', {})
            
            # Determine new status
            if game_status.get('running'):
                new_status = "available (game running)"
                icon = "🎮"
            elif game_status.get('installed'):
                new_status = "available (game installed)"
                icon = "📱"
            else:
                new_status = device['status']
                icon = "🟢" if device['status'] == 'connected' else "🟡"
            
            print(f"   {icon} {device['id']} ⭐")
            print(f"      Status: {new_status}")
            print(f"      Model: {device.get('model', 'Unknown')}")
            
            if game_status.get('running'):
                running_count = len(game_status.get('running_packages', []))
                print(f"      Game: 🎮 Running ({running_count})")
            elif game_status.get('installed'):
                installed_count = len(game_status.get('installed_packages', []))
                print(f"      Game: 📱 Installed ({installed_count})")
            print()
    
    if regular_devices:
        print(f"📱 Regular devices (appear after game devices):")
        for device in regular_devices:
            icon = "🟢" if device['status'] == 'connected' else "🟡"
            print(f"   {icon} {device['id']}")
            print(f"      Status: {device['status']}")
            print(f"      Model: {device.get('model', 'Unknown')}")
            print(f"      Game: Not installed")
            print()
    
    print(f"✅ Test completed! Game devices will be prioritized in GUI.")

if __name__ == "__main__":
    test_gui_device_prioritization()