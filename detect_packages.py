#!/usr/bin/env python3
"""
Package Detection Utility for Lineage 2M
Helps identify the correct package names on specific devices
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.core.adb_manager import ADBManager
from src.core.device_manager import DeviceManager
from src.utils.logger import get_logger
from src.utils.config import config_manager

logger = get_logger(__name__)

def scan_for_lineage_packages(device_id: str = None) -> list:
    """Scan device for any packages that might be Lineage 2M related"""
    adb = ADBManager()
    
    if device_id:
        if not adb.connect_to_device(device_id):
            print(f"❌ Could not connect to device: {device_id}")
            return []
    elif not adb.connected:
        print("❌ No device connected. Please connect a device first.")
        return []
    
    print(f"🔍 Scanning device {adb.device_id} for Lineage 2M packages...")
    
    # Search patterns for Lineage 2M
    search_patterns = [
        "lineage",
        "ncsoft",
        "l2m",
        "리니지2m",  # Korean
        "天堂2M",    # Chinese Traditional
        "天堂2m",    # Chinese Simplified
    ]
    
    found_packages = []
    
    for pattern in search_patterns:
        try:
            print(f"  Searching for pattern: '{pattern}'...")
            
            # Get all packages containing the pattern
            success, output = adb.execute_adb_command(['shell', 'pm', 'list', 'packages'])
            
            if success and output:
                lines = output.strip().split('\n')
                for line in lines:
                    if line.startswith('package:'):
                        package_name = line.replace('package:', '').strip()
                        if pattern.lower() in package_name.lower():
                            found_packages.append(package_name)
                            print(f"    ✅ Found: {package_name}")
            
        except Exception as e:
            logger.error(f"Error searching for pattern '{pattern}': {e}")
    
    # Remove duplicates and sort
    found_packages = sorted(list(set(found_packages)))
    return found_packages

def get_running_processes(device_id: str = None) -> list:
    """Get all currently running processes to identify game processes"""
    adb = ADBManager()
    
    if device_id:
        if not adb.connect_to_device(device_id):
            print(f"❌ Could not connect to device: {device_id}")
            return []
    elif not adb.connected:
        print("❌ No device connected. Please connect a device first.")
        return []
    
    print(f"🔍 Getting running processes on {adb.device_id}...")
    
    try:
        # Get all running processes
        success, output = adb.execute_adb_command(['shell', 'ps'])
        
        if not success or not output:
            # Try alternative method
            success, output = adb.execute_adb_command(['shell', 'ps', '-A'])
        
        if success and output:
            print("📋 Currently running processes:")
            lines = output.strip().split('\n')
            
            game_processes = []
            for line in lines[1:]:  # Skip header
                if any(keyword in line.lower() for keyword in ['lineage', 'ncsoft', 'l2m']):
                    game_processes.append(line.strip())
                    print(f"  🎮 {line.strip()}")
            
            return game_processes
        else:
            print("❌ Could not get process list")
            return []
            
    except Exception as e:
        logger.error(f"Error getting running processes: {e}")
        return []

def get_foreground_activity(device_id: str = None) -> str:
    """Get the current foreground activity"""
    adb = ADBManager()
    
    if device_id:
        if not adb.connect_to_device(device_id):
            print(f"❌ Could not connect to device: {device_id}")
            return None
    elif not adb.connected:
        print("❌ No device connected. Please connect a device first.")
        return None
    
    print(f"🔍 Getting foreground activity on {adb.device_id}...")
    
    try:
        # Method 1: dumpsys activity
        success, output = adb.execute_adb_command(['shell', 'dumpsys', 'activity', 'activities'])
        
        if success and output:
            lines = output.split('\n')
            for line in lines:
                if 'mResumedActivity' in line or 'mCurrentFocus' in line:
                    print(f"  🎯 Current focus: {line.strip()}")
                    return line.strip()
        
        # Method 2: dumpsys window
        success, output = adb.execute_adb_command(['shell', 'dumpsys', 'window', 'windows'])
        
        if success and output:
            lines = output.split('\n')
            for line in lines:
                if 'mCurrentFocus' in line:
                    print(f"  🎯 Current focus: {line.strip()}")
                    return line.strip()
                    
        return None
        
    except Exception as e:
        logger.error(f"Error getting foreground activity: {e}")
        return None

def main():
    """Main function to detect Lineage 2M packages"""
    print("=" * 60)
    print("🔍 Lineage 2M Package Detection Utility")
    print("=" * 60)
    
    # Initialize device manager
    device_manager = DeviceManager()
    devices = device_manager.discover_devices()
    
    if not devices:
        print("❌ No devices found!")
        return
    
    print(f"📱 Found {len(devices)} device(s):")
    for i, device in enumerate(devices, 1):
        status_icon = "🟢" if device['status'] == 'connected' else "🟡"
        print(f"  {i}. {status_icon} {device['id']} ({device['model']})")
    
    # Find SM-S908E or let user select
    target_device = None
    for device in devices:
        if "SM-S908E" in device.get('model', ''):
            target_device = device
            print(f"\n🎯 Found your SM-S908E device: {device['id']}")
            break
    
    if not target_device:
        print(f"\n❓ SM-S908E not found. Please select a device:")
        try:
            choice = int(input(f"Enter device number (1-{len(devices)}): ")) - 1
            if 0 <= choice < len(devices):
                target_device = devices[choice]
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Invalid input")
            return
    
    device_id = target_device['id']
    
    print(f"\n🔍 Analyzing device: {device_id}")
    print("-" * 40)
    
    # 1. Scan for Lineage 2M packages
    print("\n1️⃣ Scanning for Lineage 2M packages...")
    packages = scan_for_lineage_packages(device_id)
    
    if packages:
        print(f"\n✅ Found {len(packages)} potential Lineage 2M package(s):")
        for pkg in packages:
            print(f"  📦 {pkg}")
    else:
        print("❌ No Lineage 2M packages found")
    
    # 2. Check running processes
    print("\n2️⃣ Checking running processes...")
    processes = get_running_processes(device_id)
    
    # 3. Get foreground activity
    print("\n3️⃣ Getting foreground activity...")
    activity = get_foreground_activity(device_id)
    
    # 4. Generate updated config
    if packages:
        print("\n4️⃣ Suggested config update:")
        print("-" * 30)
        print("Add these packages to your bot_config.yaml:")
        print("game:")
        print("  packages:")
        for pkg in packages:
            print(f"    - \"{pkg}\"")
        
        # Update config file
        config = config_manager.get_config()
        current_packages = set(config.game.packages)
        new_packages = set(packages)
        
        if not new_packages.issubset(current_packages):
            print(f"\n💡 Updating configuration file...")
            all_packages = sorted(list(current_packages.union(new_packages)))
            
            # Read current config
            with open(config_manager.config_file, 'r') as f:
                config_data = f.read()
            
            # Update packages section
            import yaml
            config_dict = yaml.safe_load(config_data)
            config_dict['game']['packages'] = all_packages
            
            # Write updated config
            with open(config_manager.config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            print(f"✅ Updated {config_manager.config_file}")
            print(f"📦 Now monitoring {len(all_packages)} package(s)")
        else:
            print("✅ All found packages already in configuration")
    
    print(f"\n🎉 Package detection complete!")

if __name__ == "__main__":
    main()