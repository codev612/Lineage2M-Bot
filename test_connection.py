"""
Quick test script to verify ADB connection and basic functionality
"""

from adb_manager import ADBManager, GameDetector
import time

def test_adb_connection():
    print("🔧 Testing ADB Connection...")
    print("-" * 40)
    
    adb = ADBManager()
    
    # Test 1: Check if ADB is available
    print("1. Checking ADB availability...")
    if adb.check_adb_available():
        print("   ✅ ADB is available")
    else:
        print("   ❌ ADB not found in PATH")
        return False
    
    # Test 2: Discover all available devices
    print("2. Discovering all available devices...")
    devices = adb.get_all_available_devices()
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
        if adb.connect_to_device(first_device['id']):
            print(f"   ✅ Connected to: {adb.device_id}")
        else:
            print("   ❌ Failed to connect to device")
            return False
    else:
        print("   ❌ No devices available to connect to")
        print("   💡 Make sure BlueStacks or another emulator is running")
        return False
    
    # Test 4: Get device info
    print("4. Getting device information...")
    device_info = adb.get_device_info()
    for key, value in device_info.items():
        print(f"   📱 {key}: {value}")
    
    # Test 5: Get foreground app
    print("5. Checking foreground application...")
    foreground_app = adb.get_foreground_app()
    if foreground_app:
        print(f"   📱 Current app: {foreground_app}")
    else:
        print("   ⚠️  Could not detect foreground app")
    
    # Test 6: Take screenshot
    print("6. Testing screenshot capability...")
    screenshot = adb.take_screenshot()
    if screenshot is not None:
        print(f"   ✅ Screenshot taken: {screenshot.shape}")
    else:
        print("   ❌ Failed to take screenshot")
    
    # Test 7: Game detection
    print("7. Testing Lineage 2M detection...")
    game_detector = GameDetector(adb)
    is_running, package_name = game_detector.is_lineage2m_running()
    
    if is_running:
        print(f"   🎮 Lineage 2M detected: {package_name}")
    else:
        print("   ⚠️  Lineage 2M not currently running")
    
    print("\n✅ ADB connection test completed!")
    return True

if __name__ == "__main__":
    test_adb_connection()