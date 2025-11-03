"""
Device Discovery Utility for Lineage 2M Bot
Lists all available Android devices and emulators with detailed information
"""

from adb_manager import ADBManager
import sys

def main():
    """Main device discovery function"""
    print("🔍 Android Device Discovery Tool")
    print("=" * 50)
    
    adb = ADBManager()
    
    # Check ADB availability
    print("1. Checking ADB availability...")
    if not adb.check_adb_available():
        print("   ❌ ADB not found in system PATH")
        print("   💡 Please install Android SDK Platform Tools")
        return 1
    print("   ✅ ADB is available")
    
    # Discover all devices
    print("\n2. Discovering devices...")
    devices = adb.get_all_available_devices()
    
    if not devices:
        print("   ❌ No devices found")
        print("\n   💡 Troubleshooting:")
        print("      • Make sure BlueStacks or other emulator is running")
        print("      • Enable USB debugging in emulator settings")
        print("      • Try manual connection: adb connect 127.0.0.1:5555")
        return 1
    
    # Display detailed device information
    print(f"\n✅ Found {len(devices)} device(s):")
    print("=" * 80)
    
    for i, device in enumerate(devices, 1):
        status_color = "🟢" if device['status'] == 'connected' else "🟡"
        
        print(f"\n{i}. {status_color} DEVICE: {device['id']}")
        print("   " + "-" * 60)
        print(f"   📱 Type:         {device.get('type', 'Unknown')}")
        print(f"   🏭 Manufacturer: {device.get('manufacturer', 'Unknown')}")
        print(f"   📋 Model:        {device.get('model', 'Unknown')}")
        print(f"   🤖 Android:      {device.get('android_version', 'Unknown')} (API {device.get('api_level', 'Unknown')})")
        print(f"   📐 Resolution:   {device.get('resolution', 'Unknown')}")
        print(f"   ⚡ Status:       {device['status'].upper()}")
        
        if 'port' in device:
            print(f"   🔌 Port:         {device['port']}")
    
    print("\n" + "=" * 80)
    
    # Show quick connection commands
    print("\n🚀 Quick Connection Commands:")
    for i, device in enumerate(devices, 1):
        if device['status'] == 'available':
            print(f"   {i}. adb connect {device['id']}")
    
    # Test connection to first device
    if devices:
        print(f"\n🧪 Testing connection to first device: {devices[0]['id']}")
        test_device = devices[0]
        
        # Set up temporary connection
        if test_device['status'] == 'connected' or adb.connect_to_device(test_device['id']):
            print("   ✅ Connection successful")
            
            # Test screenshot capability
            print("   📸 Testing screenshot...")
            screenshot = adb.take_screenshot()
            if screenshot is not None:
                print(f"   ✅ Screenshot working: {screenshot.shape}")
            else:
                print("   ❌ Screenshot failed")
            
            # Test foreground app detection
            print("   🎯 Testing app detection...")
            foreground_app = adb.get_foreground_app()
            if foreground_app:
                print(f"   ✅ Current app: {foreground_app}")
            else:
                print("   ⚠️  Could not detect current app")
        else:
            print("   ❌ Connection test failed")
    
    print(f"\n✅ Device discovery completed!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Device discovery interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during device discovery: {e}")
        sys.exit(1)