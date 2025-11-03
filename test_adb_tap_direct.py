#!/usr/bin/env python3
"""
Direct ADB command testing for tap functionality
This script tests ADB tap commands directly to diagnose issues
"""

import subprocess
import sys
import time
from pathlib import Path

def run_adb_command(device_id, command_parts):
    """Run ADB command directly and return result"""
    full_command = ['adb', '-s', device_id] + command_parts
    print(f"\n{'='*60}")
    print(f"Executing: {' '.join(full_command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Success: {result.returncode == 0}")
        
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out")
        return False, "", "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, "", str(e)

def test_device_connection(device_id):
    """Test if device is connected"""
    print(f"\n{'='*60}")
    print(f"Testing connection to device: {device_id}")
    print(f"{'='*60}")
    
    success, stdout, stderr = run_adb_command(device_id, ['shell', 'echo', 'test'])
    if success:
        print(f"[OK] Device is connected and responding")
        return True
    else:
        print(f"[FAIL] Device connection failed")
        return False

def get_device_info(device_id):
    """Get device information"""
    print(f"\n{'='*60}")
    print(f"Getting device information for: {device_id}")
    print(f"{'='*60}")
    
    # Get device model
    print("\n1. Device Model:")
    run_adb_command(device_id, ['shell', 'getprop', 'ro.product.model'])
    
    # Get Android version
    print("\n2. Android Version:")
    run_adb_command(device_id, ['shell', 'getprop', 'ro.build.version.release'])
    
    # Get screen resolution
    print("\n3. Screen Resolution:")
    run_adb_command(device_id, ['shell', 'wm', 'size'])
    
    # Get screen density
    print("\n4. Screen Density:")
    run_adb_command(device_id, ['shell', 'wm', 'density'])
    
    # Get current foreground app
    print("\n5. Current Foreground App:")
    run_adb_command(device_id, ['shell', 'dumpsys', 'window', 'windows', '|', 'grep', '-E', 'mCurrentFocus|mFocusedApp'])

def test_tap_command(device_id, x, y):
    """Test tap command with different methods"""
    print(f"\n{'='*60}")
    print(f"Testing TAP command at coordinates ({x}, {y})")
    print(f"{'='*60}")
    
    # Method 1: Standard input tap
    print("\n--- Method 1: Standard input tap ---")
    success1, stdout1, stderr1 = run_adb_command(device_id, ['shell', 'input', 'tap', str(x), str(y)])
    
    if not success1:
        print(f"[INFO] Standard tap failed, trying alternative...")
        # Method 2: Input tap with explicit shell
        print("\n--- Method 2: Alternative tap command ---")
        success2, stdout2, stderr2 = run_adb_command(device_id, ['shell', 'input', 'tap', str(x), str(y)])
    else:
        success2 = False
    
    return success1 or success2

def test_swipe_command(device_id, x1, y1, x2, y2):
    """Test swipe command"""
    print(f"\n{'='*60}")
    print(f"Testing SWIPE command from ({x1}, {y1}) to ({x2}, {y2})")
    print(f"{'='*60}")
    
    success, stdout, stderr = run_adb_command(
        device_id, 
        ['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), '300']
    )
    return success

def test_screenshot(device_id):
    """Test screenshot capture"""
    print(f"\n{'='*60}")
    print(f"Testing screenshot capture")
    print(f"{'='*60}")
    
    # Save screenshot to device
    success1, stdout1, stderr1 = run_adb_command(
        device_id, 
        ['shell', 'screencap', '-p', '/sdcard/test_screenshot.png']
    )
    
    if success1:
        # Pull screenshot
        print("\nPulling screenshot...")
        success2, stdout2, stderr2 = run_adb_command(
            device_id,
            ['pull', '/sdcard/test_screenshot.png', 'test_screenshot.png']
        )
        if success2:
            print("[OK] Screenshot saved to test_screenshot.png")
        else:
            print("[FAIL] Failed to pull screenshot")
    
    return success1

def main():
    """Main test function"""
    print("="*60)
    print("ADB Direct Command Testing")
    print("="*60)
    
    # Get device ID from command line or list devices
    if len(sys.argv) > 1:
        device_id = sys.argv[1]
    else:
        print("\nNo device ID provided, listing available devices...")
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        print(result.stdout)
        
        # Extract device IDs
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        devices = []
        for line in lines:
            if line.strip() and 'device' in line:
                device_id = line.split()[0]
                devices.append(device_id)
        
        if not devices:
            print("[FAIL] No devices found. Please connect a device or emulator.")
            return 1
        
        if len(devices) == 1:
            device_id = devices[0]
            print(f"\n[OK] Using device: {device_id}")
        else:
            print(f"\nMultiple devices found: {devices}")
            device_id = devices[0]
            print(f"Using first device: {device_id}")
    
    # Test connection
    if not test_device_connection(device_id):
        print("\n[FAIL] Cannot connect to device. Exiting.")
        return 1
    
    # Get device info
    get_device_info(device_id)
    
    # Test screenshot
    print("\n" + "="*60)
    print("TESTING SCREENSHOT CAPTURE")
    print("="*60)
    test_screenshot(device_id)
    
    # Test tap commands
    print("\n" + "="*60)
    print("TESTING TAP COMMANDS")
    print("="*60)
    
    # Test center tap
    print("\n[WARNING] About to tap at center of screen!")
    print("Press Enter to continue, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0
    
    print("\n1. Testing center tap (540, 960)...")
    time.sleep(1)
    tap_success = test_tap_command(device_id, 540, 960)
    
    if tap_success:
        print("\n[OK] Tap command executed successfully")
        print("[INFO] Check your device screen to see if tap occurred!")
        time.sleep(2)
    else:
        print("\n[FAIL] Tap command failed")
        print("Check the error messages above for details")
    
    # Test swipe
    print("\n" + "="*60)
    print("TESTING SWIPE COMMAND")
    print("="*60)
    print("\n[WARNING] About to swipe right!")
    print("Press Enter to continue, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0
    
    swipe_success = test_swipe_command(device_id, 300, 960, 780, 960)
    
    if swipe_success:
        print("\n[OK] Swipe command executed successfully")
        print("[INFO] Check your device screen to see if swipe occurred!")
    else:
        print("\n[FAIL] Swipe command failed")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Device ID: {device_id}")
    print(f"Tap test: {'[PASSED]' if tap_success else '[FAILED]'}")
    print(f"Swipe test: {'[PASSED]' if swipe_success else '[FAILED]'}")
    
    if not tap_success:
        print("\n[TIPS] TROUBLESHOOTING:")
        print("1. Make sure device screen is unlocked")
        print("2. Check if app has permission to use input")
        print("3. Try different coordinates based on device resolution")
        print("4. Verify device is connected: adb devices")
        print("5. Check if device requires special permissions")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(0)

