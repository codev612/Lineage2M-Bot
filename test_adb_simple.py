#!/usr/bin/env python3
"""
Simple ADB tap test - runs commands directly without prompts
"""

import subprocess
import sys
import time

def run_adb(device_id, cmd_parts):
    """Run ADB command and show result"""
    full_cmd = ['adb', '-s', device_id] + cmd_parts
    print(f"\nCommand: {' '.join(full_cmd)}")
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    
    print("="*60)
    print("ADB Direct Tap Test")
    print("="*60)
    print(f"Device: {device_id}")
    
    # Get resolution
    print("\n1. Getting device resolution...")
    run_adb(device_id, ['shell', 'wm', 'size'])
    
    # Test tap
    print("\n2. Testing tap at center (540, 960)...")
    time.sleep(1)
    success = run_adb(device_id, ['shell', 'input', 'tap', '540', '960'])
    
    if success:
        print("\n[OK] Tap command executed successfully!")
        print("Check your device screen - tap should have occurred at center")
    else:
        print("\n[FAIL] Tap command failed!")
        print("Check error messages above")
    
    time.sleep(2)
    
    # Test alternative coordinates (center of 1920x1080 = 960, 540)
    print("\n3. Testing tap at center (960, 540) - adjusted for 1920x1080...")
    time.sleep(1)
    success2 = run_adb(device_id, ['shell', 'input', 'tap', '960', '540'])
    
    if success2:
        print("\n[OK] Alternative tap executed successfully!")
    else:
        print("\n[FAIL] Alternative tap failed!")
    
    # Test swipe
    print("\n4. Testing swipe...")
    time.sleep(1)
    success3 = run_adb(device_id, ['shell', 'input', 'swipe', '300', '540', '700', '540', '300'])
    
    if success3:
        print("\n[OK] Swipe executed successfully!")
    else:
        print("\n[FAIL] Swipe failed!")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    print("\nNOTE: If taps don't appear on screen:")
    print("1. Make sure screen is unlocked")
    print("2. Check if app is in foreground")
    print("3. Verify coordinates match device resolution")
    print("4. Try different coordinates")

if __name__ == "__main__":
    main()

