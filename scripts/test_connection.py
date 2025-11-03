#!/usr/bin/env python3
"""
Connection Test Script - Test ADB connection and basic functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.device_manager import DeviceManager
from src.modules.game_detector import GameDetector
from src.utils.config import config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_adb_connection():
    """Test ADB connection and functionality"""
    print("🔧 Testing ADB Connection...")
    print("-" * 40)
    
    try:
        device_manager = DeviceManager()
        config = config_manager.get_config()
        
        # Test 1: Check ADB availability
        print("1. Checking ADB availability...")
        if device_manager.adb.check_adb_available():
            print("   ✅ ADB is available")
        else:
            print("   ❌ ADB not found in PATH")
            return False
        
        # Test 2: Discover devices
        print("2. Discovering all available devices...")
        devices = device_manager.discover_devices()
        if devices:
            print(f"   ✅ Found {len(devices)} device(s):")
            for i, device in enumerate(devices, 1):
                status_icon = "🟢" if device['status'] == 'connected' else "🟡"
                print(f"      {i}. {status_icon} {device['id']} ({device['type']})")
        else:
            print("   ⚠️  No devices found")
        
        # Test 3: Connect to first available device
        print("3. Connecting to first available device...")
        if devices:
            first_device = devices[0]
            if device_manager.select_device_by_id(first_device['id']):
                if device_manager.connect_to_selected_device():
                    print(f"   ✅ Connected to: {device_manager.get_current_device_id()}")
                else:
                    print("   ❌ Failed to connect to device")
                    return False
            else:
                print("   ❌ Failed to select device")
                return False
        else:
            print("   ❌ No devices available to connect to")
            print("   💡 Make sure BlueStacks or another emulator is running")
            return False
        
        # Test 4: Get device info
        print("4. Getting device information...")
        device_info = device_manager.get_selected_device_info()
        for key, value in device_info.items():
            if key in ['model', 'android_version', 'resolution', 'type']:
                print(f"   📱 {key}: {value}")
        
        # Test 5: Get foreground app
        print("5. Checking foreground application...")
        foreground_app = device_manager.adb.get_foreground_app()
        if foreground_app:
            print(f"   📱 Current app: {foreground_app}")
        else:
            print("   ⚠️  Could not detect foreground app")
        
        # Test 6: Take screenshot
        print("6. Testing screenshot capability...")
        screenshot = device_manager.adb.take_screenshot()
        if screenshot is not None:
            print(f"   ✅ Screenshot taken: {screenshot.shape}")
        else:
            print("   ❌ Screenshot failed")
        
        # Test 7: Game detection
        print("7. Testing Lineage 2M detection...")
        game_detector = GameDetector(device_manager.adb, config.game)
        is_running, package_name = game_detector.is_lineage2m_running()
        
        if is_running:
            print(f"   🎮 Lineage 2M detected: {package_name}")
            
            # Test game state detection
            game_state = game_detector.detect_game_state()
            if game_state.get('screenshot_taken'):
                print(f"   ✅ Game state detection working")
                print(f"   📊 UI elements detected: {game_state.get('ui_elements', 0)}")
                print(f"   🎨 Menu state: {game_state.get('menu_state', 'unknown')}")
            else:
                print(f"   ⚠️  Game state detection issues")
        else:
            print("   ⚠️  Lineage 2M not currently running")
            
            # Check for installed packages
            installed_packages = game_detector.get_installed_lineage2m_packages()
            if installed_packages:
                print(f"   📦 Found installed packages: {installed_packages}")
            else:
                print("   📦 No Lineage 2M packages found")
        
        print("\n✅ ADB connection test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during connection test: {e}")
        logger.exception("Connection test error")
        return False

if __name__ == "__main__":
    success = test_adb_connection()
    exit(0 if success else 1)