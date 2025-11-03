#!/usr/bin/env python3
"""
Quick test to check game status on SM-S908E device
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.core.adb_manager import ADBManager
from src.utils.config import config_manager

def test_device_game_status():
    """Test game status on specific device"""
    
    # Force reload config
    config_manager._load_config()  # Force reload
    config = config_manager.get_config()
    game_packages = config.game.packages
    
    print("🧪 Testing Game Status Detection")
    print("=" * 40)
    print(f"Looking for packages: {game_packages}")
    print(f"Total packages to check: {len(game_packages)}")
    
    # Connect to SM-S908E device
    adb = ADBManager()
    device_id = "127.0.0.1:5555"  # Your SM-S908E
    
    print(f"\n🔗 Connecting to {device_id}...")
    
    if not adb.connect_to_device(device_id):
        print(f"❌ Failed to connect to {device_id}")
        return
    
    print(f"✅ Connected to {device_id}")
    
    # Test game status
    print(f"\n🎮 Checking game status...")
    game_status = adb.check_game_status(device_id, game_packages)
    
    print(f"📊 Game Status Results:")
    print(f"  Installed: {game_status['game_installed']}")
    print(f"  Running: {game_status['game_running']}")
    print(f"  Installed packages: {game_status['installed_packages']}")
    print(f"  Running packages: {game_status['running_packages']}")
    print(f"  Foreground package: {game_status['foreground_package']}")
    
    # Manual check for the specific package we found
    print(f"\n🔍 Manual check for 'com.ncsoft.lineage2mnu'...")
    
    # Check if installed
    success, output = adb.execute_adb_command(['-s', device_id, 'shell', 'pm', 'list', 'packages', 'com.ncsoft.lineage2mnu'])
    print(f"  Package check: {'✅' if success and 'com.ncsoft.lineage2mnu' in output else '❌'}")
    if success:
        print(f"  Output: {output.strip()}")
    
    # Check if running
    success, output = adb.execute_adb_command(['-s', device_id, 'shell', 'pidof', 'com.ncsoft.lineage2mnu'])
    print(f"  Process check: {'✅' if success and output.strip() else '❌'}")
    if success:
        print(f"  PID: {output.strip()}")
    
    # Check foreground
    success, output = adb.execute_adb_command(['-s', device_id, 'shell', 'dumpsys', 'activity', 'activities'])
    if success:
        lines = output.split('\n')
        for line in lines:
            if 'mResumedActivity' in line and 'lineage2mnu' in line:
                print(f"  Foreground: ✅ {line.strip()}")
                break
        else:
            print(f"  Foreground: ❌ Not in foreground")

if __name__ == "__main__":
    test_device_game_status()