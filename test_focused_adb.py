#!/usr/bin/env python3
"""
Focused test of ADB game status with debug output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.adb_manager import ADBManager
from src.utils.config import config_manager

def main():
    print("🧪 Focused ADB game status test...")
    
    # Get packages from config
    game_config = config_manager.get_game_config()
    packages = game_config.packages
    print(f"📦 Config packages: {packages}")
    
    # Test ADB manager
    adb = ADBManager()
    device_id = "127.0.0.1:5555"
    
    print(f"\n🔗 Testing device: {device_id}")
    
    # Connect
    success = adb.connect_to_device(device_id)
    print(f"   📱 Connection: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        # Test manual ADB command first
        cmd = ['-s', device_id, 'shell', 'pm', 'list', 'packages']
        success_manual, output_manual = adb.execute_adb_command(cmd)
        
        print(f"   🔧 Manual command success: {success_manual}")
        print(f"   📄 Manual output (first few lines):")
        if output_manual:
            lines = output_manual.strip().split('\n')[:5]
            for line in lines:
                print(f"      {line}")
            print(f"      ... (total {len(output_manual.split())} lines)")
        
        # Test with each package individually
        for package in packages:
            print(f"\n   🧪 Testing package: {package}")
            cmd = ['-s', device_id, 'shell', 'pm', 'list', 'packages', package]
            success_pkg, output_pkg = adb.execute_adb_command(cmd)
            print(f"      Command: adb -s {device_id} shell pm list packages {package}")
            print(f"      Success: {success_pkg}")
            print(f"      Output: '{output_pkg.strip()}' (contains package: {'package:' + package in output_pkg})")
        
        # Now test the check_game_status method with explicit packages
        print(f"\n   🎮 Testing check_game_status with packages...")
        status = adb.check_game_status(device_id, packages)
        print(f"   📊 Status: {status}")

if __name__ == "__main__":
    main()