"""
Quick Device Connection Utility
Allows quick connection to a specific device by ID or interactive selection
"""

from adb_manager import ADBManager
import sys

def connect_to_device(device_id=None):
    """Connect to a specific device or show selection menu"""
    adb = ADBManager()
    
    if not adb.check_adb_available():
        print("❌ ADB not available. Please install Android SDK Platform Tools.")
        return False
    
    if device_id:
        # Connect to specific device
        print(f"🔗 Connecting to {device_id}...")
        if adb.connect_to_device(device_id):
            print(f"✅ Successfully connected to {device_id}")
            
            # Show device info
            device_info = adb.get_device_detailed_info(device_id)
            print(f"📱 Device: {device_info['model']} ({device_info['type']})")
            print(f"🤖 Android: {device_info['android_version']}")
            print(f"📐 Resolution: {device_info['resolution']}")
            return True
        else:
            print(f"❌ Failed to connect to {device_id}")
            return False
    else:
        # Show device selection menu
        devices = adb.get_all_available_devices()
        
        if not devices:
            print("❌ No devices found!")
            return False
        
        print(f"📱 Available devices:")
        for i, device in enumerate(devices, 1):
            status_icon = "🟢" if device['status'] == 'connected' else "🟡"
            print(f"  {i}. {status_icon} {device['id']} ({device['type']})")
        
        try:
            choice = input(f"\nSelect device (1-{len(devices)}): ").strip()
            device_index = int(choice) - 1
            
            if 0 <= device_index < len(devices):
                selected_device = devices[device_index]
                return connect_to_device(selected_device['id'])
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(devices)}")
                return False
                
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled by user")
            return False

def main():
    """Main function"""
    print("🔗 Quick Device Connection Utility")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # Device ID provided as argument
        device_id = sys.argv[1]
        success = connect_to_device(device_id)
    else:
        # Interactive selection
        success = connect_to_device()
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        sys.exit(0)