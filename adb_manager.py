"""
Lineage 2M Bot - ADB Connection Manager
Handles connection to Android emulator and basic game detection
"""

import subprocess
import time
import re
import logging
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADBManager:
    def __init__(self):
        self.device_id: Optional[str] = None
        self.connected = False
        
    def check_adb_available(self) -> bool:
        """Check if ADB is available in system PATH"""
        try:
            result = subprocess.run(['adb', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"ADB available: {result.stdout.strip().split()[0]}")
                return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"ADB not found in PATH: {e}")
            return False
    
    def get_connected_devices(self) -> List[str]:
        """Get list of connected Android devices/emulators"""
        try:
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                devices = []
                for line in lines:
                    if line.strip() and 'device' in line:
                        device_id = line.split()[0]
                        devices.append(device_id)
                return devices
            return []
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout getting devices: {e}")
            return []
    
    def get_all_available_devices(self) -> List[Dict]:
        """Get comprehensive list of all available devices with detailed information"""
        devices = []
        
        # Get already connected devices
        connected_devices = self.get_connected_devices()
        
        for device_id in connected_devices:
            device_info = self.get_device_detailed_info(device_id)
            device_info['id'] = device_id
            device_info['status'] = 'connected'
            devices.append(device_info)
        
        # Try to discover BlueStacks instances
        bluestacks_devices = self.discover_bluestacks_devices()
        for bs_device in bluestacks_devices:
            # Check if already in connected devices
            if bs_device['id'] not in connected_devices:
                bs_device['status'] = 'available'
                devices.append(bs_device)
        
        return devices
    
    def discover_bluestacks_devices(self) -> List[Dict]:
        """Discover potential BlueStacks emulator instances"""
        bluestacks_devices = []
        bluestacks_ports = [5555, 5554, 5556, 5558, 5562, 5564, 5566, 5568]
        
        logger.info("Scanning for BlueStacks instances...")
        
        for port in bluestacks_ports:
            device_id = f"127.0.0.1:{port}"
            
            # Try to connect temporarily to test
            try:
                result = subprocess.run(['adb', 'connect', device_id], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    # Get device info if connection successful
                    device_info = self.get_device_detailed_info(device_id)
                    device_info['id'] = device_id
                    device_info['type'] = 'BlueStacks'
                    device_info['port'] = port
                    bluestacks_devices.append(device_info)
                    
            except (subprocess.TimeoutExpired, Exception) as e:
                # Port not available or no emulator running
                continue
        
        return bluestacks_devices
    
    def get_device_detailed_info(self, device_id: str) -> Dict:
        """Get detailed information about a specific device"""
        info = {
            'id': device_id,
            'model': 'Unknown',
            'android_version': 'Unknown',
            'resolution': 'Unknown',
            'manufacturer': 'Unknown',
            'type': 'Unknown',
            'api_level': 'Unknown'
        }
        
        try:
            # Get model
            result = subprocess.run(['adb', '-s', device_id, 'shell', 'getprop', 'ro.product.model'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['model'] = result.stdout.strip()
            
            # Get manufacturer
            result = subprocess.run(['adb', '-s', device_id, 'shell', 'getprop', 'ro.product.manufacturer'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['manufacturer'] = result.stdout.strip()
            
            # Get Android version
            result = subprocess.run(['adb', '-s', device_id, 'shell', 'getprop', 'ro.build.version.release'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['android_version'] = result.stdout.strip()
            
            # Get API level
            result = subprocess.run(['adb', '-s', device_id, 'shell', 'getprop', 'ro.build.version.sdk'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['api_level'] = result.stdout.strip()
            
            # Get screen resolution
            result = subprocess.run(['adb', '-s', device_id, 'shell', 'wm', 'size'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'Physical size:' in result.stdout:
                resolution = result.stdout.split('Physical size:')[1].strip()
                info['resolution'] = resolution
            
            # Determine device type
            if '127.0.0.1:' in device_id:
                if 'BlueStacks' in info.get('model', '') or 'bstk' in info.get('model', '').lower():
                    info['type'] = 'BlueStacks'
                else:
                    info['type'] = 'Emulator'
            elif 'emulator-' in device_id:
                info['type'] = 'Android Studio Emulator'
            else:
                info['type'] = 'Physical Device'
                    
        except Exception as e:
            logger.warning(f"Could not get detailed info for {device_id}: {e}")
        
        return info
    
    def connect_to_bluestacks(self) -> bool:
        """Connect to BlueStacks emulator (common ports: 5555, 5554, 5556, 5558)"""
        bluestacks_ports = [5555, 5554, 5556, 5558, 5562, 5564]
        
        for port in bluestacks_ports:
            try:
                connect_address = f"127.0.0.1:{port}"
                logger.info(f"Attempting to connect to BlueStacks on {connect_address}")
                
                result = subprocess.run(['adb', 'connect', connect_address], 
                                      capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0 and 'connected' in result.stdout.lower():
                    self.device_id = connect_address
                    self.connected = True
                    logger.info(f"Successfully connected to BlueStacks: {connect_address}")
                    return True
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout connecting to port {port}")
                continue
        
        return False
    
    def connect_to_device(self, device_id: str = None) -> bool:
        """Connect to a specific device or auto-detect"""
        if not self.check_adb_available():
            return False
        
        # If device_id not specified, try to auto-connect
        if not device_id:
            # First try to connect to BlueStacks
            if self.connect_to_bluestacks():
                return True
            
            # Then check for already connected devices
            devices = self.get_connected_devices()
            if devices:
                self.device_id = devices[0]
                self.connected = True
                logger.info(f"Using already connected device: {self.device_id}")
                return True
            else:
                logger.error("No devices found. Make sure your emulator is running.")
                return False
        else:
            self.device_id = device_id
            self.connected = True
            return True
    
    def execute_adb_command(self, command: List[str]) -> Tuple[bool, str]:
        """Execute ADB command and return success status and output"""
        if not self.connected or not self.device_id:
            return False, "Not connected to any device"
        
        full_command = ['adb', '-s', self.device_id] + command
        try:
            result = subprocess.run(full_command, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timeout"
    
    def get_running_packages(self) -> List[str]:
        """Get list of currently running packages"""
        success, output = self.execute_adb_command(['shell', 'pm', 'list', 'packages', '-3'])
        if success:
            packages = []
            for line in output.strip().split('\n'):
                if line.startswith('package:'):
                    package_name = line.replace('package:', '')
                    packages.append(package_name)
            return packages
        return []
    
    def is_app_running(self, package_name: str) -> bool:
        """Check if a specific app is currently running"""
        success, output = self.execute_adb_command(['shell', 'pidof', package_name])
        return success and output.strip() != ""
    
    def get_foreground_app(self) -> Optional[str]:
        """Get the package name of the currently foreground app"""
        try:
            # Method 1: Use dumpsys activity for current activity
            success, output = self.execute_adb_command([
                'shell', 'dumpsys', 'activity', 'activities'
            ])
            
            if success and output:
                # Look for mResumedActivity or mCurrentFocus
                lines = output.split('\n')
                for line in lines:
                    if 'mResumedActivity' in line or 'mCurrentFocus' in line:
                        # Extract package name
                        match = re.search(r'([a-zA-Z0-9._]+)/[a-zA-Z0-9._]+', line)
                        if match:
                            return match.group(1)
            
            # Method 2: Use top activity (newer Android versions)
            success, output = self.execute_adb_command([
                'shell', 'dumpsys', 'activity', 'top'
            ])
            
            if success and output:
                lines = output.split('\n')
                for line in lines:
                    if 'ACTIVITY' in line and '/' in line:
                        match = re.search(r'([a-zA-Z0-9._]+)/[a-zA-Z0-9._]+', line)
                        if match:
                            return match.group(1)
            
            # Method 3: Simple ps approach
            success, output = self.execute_adb_command([
                'shell', 'ps', '-A', '-o', 'NAME'
            ])
            
            if success and output:
                # This gives us running processes, we can check the most recent UI process
                lines = output.strip().split('\n')
                for line in reversed(lines[1:]):  # Skip header
                    if 'com.' in line and not any(sys in line for sys in ['system', 'android']):
                        return line.strip()
        
        except Exception as e:
            logger.error(f"Error getting foreground app: {e}")
        
        return None
    
    def take_screenshot(self) -> Optional[np.ndarray]:
        """Take screenshot and return as OpenCV image"""
        if not self.connected or not self.device_id:
            return None
        
        try:
            # Use shell command to save screenshot to device, then pull it
            temp_path = '/sdcard/screenshot_temp.png'
            
            # Take screenshot and save to device
            cmd1 = ['adb', '-s', self.device_id, 'shell', 'screencap', '-p', temp_path]
            result1 = subprocess.run(cmd1, capture_output=True, timeout=10)
            
            if result1.returncode != 0:
                logger.error("Failed to take screenshot on device")
                return None
            
            # Pull screenshot from device
            cmd2 = ['adb', '-s', self.device_id, 'pull', temp_path, 'temp_screenshot.png']
            result2 = subprocess.run(cmd2, capture_output=True, timeout=10)
            
            if result2.returncode != 0:
                logger.error("Failed to pull screenshot from device")
                return None
            
            # Load image with OpenCV
            opencv_image = cv2.imread('temp_screenshot.png')
            
            # Clean up temp files
            subprocess.run(['adb', '-s', self.device_id, 'shell', 'rm', temp_path], 
                         capture_output=True, timeout=5)
            
            import os
            if os.path.exists('temp_screenshot.png'):
                os.remove('temp_screenshot.png')
            
            return opencv_image
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None
    
    def get_device_info(self) -> dict:
        """Get device information"""
        info = {}
        
        # Get device model
        success, output = self.execute_adb_command(['shell', 'getprop', 'ro.product.model'])
        if success:
            info['model'] = output.strip()
        
        # Get Android version
        success, output = self.execute_adb_command(['shell', 'getprop', 'ro.build.version.release'])
        if success:
            info['android_version'] = output.strip()
        
        # Get screen resolution
        success, output = self.execute_adb_command(['shell', 'wm', 'size'])
        if success and 'Physical size:' in output:
            resolution = output.split('Physical size:')[1].strip()
            info['resolution'] = resolution
        
        return info

class GameDetector:
    def __init__(self, adb_manager: ADBManager):
        self.adb = adb_manager
        # Common Lineage 2M package names (may vary by region)
        self.lineage2m_packages = [
            'com.ncsoft.lineage2m',
            'com.ncsoft.lineage2m.android',
            'com.ncsoft.lineage2m.global',
            'com.ncsoft.lineage2m.sea',
            'com.ncsoft.lineage2m.kr'
        ]
    
    def is_lineage2m_running(self) -> Tuple[bool, Optional[str]]:
        """Check if Lineage 2M is running and return package name if found"""
        foreground_app = self.adb.get_foreground_app()
        
        # Check if foreground app is Lineage 2M
        if foreground_app:
            for package in self.lineage2m_packages:
                if package in foreground_app:
                    return True, package
        
        # Check if any Lineage 2M package is running (even if not foreground)
        for package in self.lineage2m_packages:
            if self.adb.is_app_running(package):
                return True, package
        
        return False, None
    
    def detect_game_state(self) -> dict:
        """Detect current game state using screenshot analysis"""
        screenshot = self.adb.take_screenshot()
        if screenshot is None:
            return {'status': 'error', 'message': 'Could not take screenshot'}
        
        # Basic game state detection (this would be expanded based on actual game UI)
        state = {
            'status': 'unknown',
            'screenshot_taken': True,
            'screen_size': screenshot.shape[:2],
            'message': 'Screenshot captured successfully'
        }
        
        # Here you would add specific game state detection logic
        # For example, looking for specific UI elements, colors, text, etc.
        
        return state

def main():
    """Main function to test ADB connection and game detection"""
    logger.info("Starting Lineage 2M Bot - ADB Connection Test")
    
    # Initialize ADB manager
    adb = ADBManager()
    
    # Connect to device
    if not adb.connect_to_device():
        logger.error("Failed to connect to any device. Make sure your emulator is running.")
        return
    
    # Get device info
    device_info = adb.get_device_info()
    logger.info(f"Connected to device: {device_info}")
    
    # Initialize game detector
    game_detector = GameDetector(adb)
    
    # Check if Lineage 2M is running
    is_running, package_name = game_detector.is_lineage2m_running()
    
    if is_running:
        logger.info(f"Lineage 2M is running! Package: {package_name}")
        
        # Get current game state
        game_state = game_detector.detect_game_state()
        logger.info(f"Game state: {game_state}")
        
    else:
        logger.info("Lineage 2M is not currently running.")
        
        # Show currently running apps
        foreground_app = adb.get_foreground_app()
        if foreground_app:
            logger.info(f"Current foreground app: {foreground_app}")
        
        # List all installed packages (for debugging)
        packages = adb.get_running_packages()
        logger.info(f"Found {len(packages)} installed packages")

if __name__ == "__main__":
    main()