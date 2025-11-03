#!/usr/bin/env python3
"""
Debug script to test configuration loading and game detection
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import yaml
from src.core.adb_manager import ADBManager
from src.utils.config import ConfigManager

def debug_config_and_detection():
    """Debug configuration loading and game detection"""
    
    print("🔧 Configuration and Game Detection Debug")
    print("=" * 50)
    
    # 1. Direct config file read
    print("1️⃣ Direct config file read:")
    config_file = Path("config/bot_config.yaml")
    with open(config_file, 'r') as f:
        direct_config = yaml.safe_load(f)
    
    direct_packages = direct_config['game']['packages']
    print(f"   📦 Direct packages: {direct_packages}")
    print(f"   ✅ Has lineage2mnu: {'com.ncsoft.lineage2mnu' in direct_packages}")
    
    # 2. ConfigManager test
    print("\n2️⃣ ConfigManager test:")
    config_manager = ConfigManager()
    managed_config = config_manager.get_config()
    managed_packages = managed_config.game.packages
    print(f"   📦 Managed packages: {managed_packages}")
    print(f"   ✅ Has lineage2mnu: {'com.ncsoft.lineage2mnu' in managed_packages}")
    
    # 3. Force reload test
    print("\n3️⃣ Force reload test:")
    config_manager._load_config()  # Force reload
    reloaded_config = config_manager.get_config()
    reloaded_packages = reloaded_config.game.packages
    print(f"   📦 Reloaded packages: {reloaded_packages}")
    print(f"   ✅ Has lineage2mnu: {'com.ncsoft.lineage2mnu' in reloaded_packages}")
    
    # 4. ADB game status test
    print("\n4️⃣ ADB game status test:")
    adb = ADBManager()
    device_id = "127.0.0.1:5555"
    
    if adb.connect_to_device(device_id):
        print(f"   🔗 Connected to {device_id}")
        
        # Test with all three package lists
        for name, packages in [
            ("Direct", direct_packages),
            ("Managed", managed_packages), 
            ("Reloaded", reloaded_packages)
        ]:
            print(f"\n   🧪 Testing {name} packages: {packages}")
            game_status = adb.check_game_status(device_id, packages)
            
            print(f"      Installed: {game_status['game_installed']}")
            print(f"      Running: {game_status['game_running']}")
            print(f"      Packages found: {game_status['installed_packages']}")
            print(f"      Running packages: {game_status['running_packages']}")
            
    else:
        print(f"   ❌ Could not connect to {device_id}")
    
    # 5. Manual package check
    print("\n5️⃣ Manual package check:")
    if adb.connected:
        success, output = adb.execute_adb_command(['-s', device_id, 'shell', 'pm', 'list', 'packages', 'com.ncsoft.lineage2mnu'])
        print(f"   Package check result: {success}")
        print(f"   Output: {output.strip() if output else 'None'}")
    
    print("\n🎉 Debug complete!")

if __name__ == "__main__":
    debug_config_and_detection()