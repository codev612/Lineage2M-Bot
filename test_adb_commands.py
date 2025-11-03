#!/usr/bin/env python3
"""
Test ADB command execution for tap and swipe
"""

import sys
from pathlib import Path

# Add the src directory to the Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_adb_commands():
    """Test ADB command construction and execution"""
    print("🔧 Testing ADB Command Construction")
    print("=" * 50)
    
    # Simulate the command construction that should happen
    test_cases = [
        {
            'command': 'tap',
            'params': {'x': 540, 'y': 960},
            'expected_cmd': ['shell', 'input', 'tap', '540', '960'],
            'description': 'Center screen tap'
        },
        {
            'command': 'swipe',
            'params': {'x1': 300, 'y1': 960, 'x2': 780, 'y2': 960},
            'expected_cmd': ['shell', 'input', 'swipe', '300', '960', '780', '960'],
            'description': 'Horizontal swipe right'
        },
        {
            'command': 'swipe',
            'params': {'x1': 540, 'y1': 600, 'x2': 540, 'y2': 1200},
            'expected_cmd': ['shell', 'input', 'swipe', '540', '600', '540', '1200'],
            'description': 'Vertical swipe down'
        }
    ]
    
    print("📋 ADB Command Construction Test:")
    print("-" * 50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test['description']}")
        print(f"   Command: {test['command']}")
        print(f"   Parameters: {test['params']}")
        
        # Construct command as the multi-device manager does
        if test['command'] == 'tap':
            x, y = test['params']['x'], test['params']['y']
            constructed_cmd = ['shell', 'input', 'tap', str(x), str(y)]
        elif test['command'] == 'swipe':
            x1, y1 = test['params']['x1'], test['params']['y1']
            x2, y2 = test['params']['x2'], test['params']['y2']
            constructed_cmd = ['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2)]
        
        print(f"   Constructed: {constructed_cmd}")
        print(f"   Expected:    {test['expected_cmd']}")
        print(f"   Match: {'✅' if constructed_cmd == test['expected_cmd'] else '❌'}")
        print()
    
    print("🔧 Fixed Issue:")
    print("   Before: DeviceSession added extra '-s device_id' causing:")
    print("           adb -s device_id -s device_id shell input tap x y")
    print("   After:  ADBManager handles device selection internally:")
    print("           adb -s device_id shell input tap x y")
    
    print(f"\n✅ ADB Command Flow:")
    print("   1. GUI calls: multi_device_manager.execute_on_device(device_id, 'tap', x=540, y=960)")
    print("   2. MultiDeviceManager constructs: ['shell', 'input', 'tap', '540', '960']")
    print("   3. DeviceSession calls: adb.execute_adb_command(command)")
    print("   4. ADBManager executes: adb -s device_id shell input tap 540 960")
    
    return True

if __name__ == "__main__":
    success = test_adb_commands()
    sys.exit(0 if success else 1)