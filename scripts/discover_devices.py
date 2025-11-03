#!/usr/bin/env python3
"""
Device Discovery Script - Discover and list all available Android devices
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.device_manager import DeviceManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Main device discovery function"""
    print("🔍 Android Device Discovery Tool")
    print("=" * 50)
    
    try:
        device_manager = DeviceManager()
        
        # Discover devices
        devices = device_manager.discover_devices()
        
        if not devices:
            print("❌ No devices found")
            print("\n💡 Troubleshooting:")
            print("   • Make sure BlueStacks or other emulator is running")
            print("   • Enable USB debugging in emulator settings")
            print("   • Try manual connection: adb connect 127.0.0.1:5555")
            return 1
        
        # Display devices
        device_manager.display_devices()
        
        # Show quick connection commands
        print("\n🚀 Quick Connection Commands:")
        for i, device in enumerate(devices, 1):
            if device['status'] == 'available':
                print(f"   {i}. adb connect {device['id']}")
        
        # Test connection to first device
        if devices:
            print(f"\n🧪 Testing connection to first device: {devices[0]['id']}")
            
            if device_manager.select_device_by_id(devices[0]['id']):
                if device_manager.connect_to_selected_device():
                    print("   ✅ Connection successful")
                    
                    # Test capabilities
                    capabilities = device_manager.get_device_capabilities()
                    print("   📋 Device capabilities:")
                    for capability, available in capabilities.items():
                        status = "✅" if available else "❌"
                        print(f"      {status} {capability}")
                else:
                    print("   ❌ Connection failed")
        
        print(f"\n✅ Device discovery completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during device discovery: {e}")
        logger.exception("Device discovery error")
        return 1

if __name__ == "__main__":
    exit(main())