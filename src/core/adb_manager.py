"""
ADB Manager - Core module for Android Debug Bridge operations
Handles connection to Android devices/emulators and basic ADB operations
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
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.exceptions import ADBError, DeviceNotFoundError, ConnectionTimeoutError

logger = get_logger(__name__)

class ADBManager:
    """
    Manages ADB connections and operations for Android devices/emulators
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize ADB Manager
        
        Args:
            timeout: Default timeout for ADB operations in seconds
        """
        self.device_id: Optional[str] = None
        self.connected = False
        self.timeout = timeout
        self._validate_adb()
    
    def _validate_adb(self) -> None:
        """Validate that ADB is available in system PATH"""
        if not self.check_adb_available():
            raise ADBError("ADB not found in system PATH. Please install Android SDK Platform Tools.")
    
    def check_adb_available(self) -> bool:
        """Check if ADB is available in system PATH"""
        try:
            result = subprocess.run(['adb', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip().split()[0]
                logger.info(f"ADB available: {version}")
                return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"ADB not found in PATH: {e}")
            return False
    
    def get_connected_devices(self) -> List[str]:
        """Get list of connected Android devices/emulators"""
        try:
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True, timeout=self.timeout)
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
            raise ConnectionTimeoutError("Timeout while getting connected devices")
    
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
                raise DeviceNotFoundError("No devices found")
        else:
            self.device_id = device_id
            self.connected = True
            return True
    
    def execute_adb_command(self, command: List[str]) -> Tuple[bool, str]:
        """Execute ADB command and return success status and output"""
        if not self.connected or not self.device_id:
            error_msg = f"Not connected - connected={self.connected}, device_id={self.device_id}"
            logger.error(f"ADBManager.execute_adb_command: {error_msg}")
            raise ADBError(error_msg)
        
        full_command = ['adb', '-s', self.device_id] + command
        logger.info(f"ADBManager executing: {' '.join(full_command)}")
        try:
            result = subprocess.run(full_command, capture_output=True, text=True, timeout=self.timeout)
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            if not success:
                logger.warning(f"ADB command failed: {' '.join(full_command)}")
                logger.warning(f"Return code: {result.returncode}, Error output: {output}")
            else:
                logger.info(f"ADB command succeeded: {' '.join(full_command)}")
                if output and len(output.strip()) > 0:
                    logger.info(f"Command output: {output.strip()[:200]}")  # First 200 chars
            
            return success, output
        except subprocess.TimeoutExpired:
            logger.error(f"ADB command timeout: {' '.join(full_command)}")
            raise ConnectionTimeoutError("ADB command timeout")
        except Exception as e:
            logger.error(f"ADB command exception: {' '.join(full_command)}, error: {e}", exc_info=True)
            raise
    
    def is_app_running(self, package_name: str) -> bool:
        """Check if a specific app is currently running"""
        try:
            # Method 1: Use pidof (works if process name matches package)
            success, output = self.execute_adb_command(['shell', 'pidof', package_name])
            if success and output.strip():
                logger.debug(f"App {package_name} found running via pidof: {output.strip()}")
                return True
            
            # Method 2: Use ps to find running processes
            success, output = self.execute_adb_command(['shell', 'ps', '-A'])
            if success and output:
                # Check if package name appears in process list
                if package_name in output:
                    logger.debug(f"App {package_name} found running via ps")
                    return True
            
            # Method 3: Use dumpsys meminfo to check if app is in memory
            success, output = self.execute_adb_command(['shell', 'dumpsys', 'meminfo', package_name])
            if success and output and 'No process found' not in output:
                logger.debug(f"App {package_name} found running via dumpsys meminfo")
                return True
            
            logger.debug(f"App {package_name} not found running")
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if app is running {package_name}: {e}")
            return False
    
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
        
        except Exception as e:
            logger.error(f"Error getting foreground app: {e}")
        
        return None
    
    def take_screenshot(self, save_path: str = None) -> Optional[np.ndarray]:
        """Take screenshot and return as OpenCV image"""
        if not self.connected or not self.device_id:
            raise ADBError("Not connected to any device")
        
        try:
            # Use shell command to save screenshot to device, then pull it
            temp_path = '/sdcard/screenshot_temp.png'
            local_path = save_path or 'temp_screenshot.png'
            
            # Take screenshot and save to device
            cmd1 = ['adb', '-s', self.device_id, 'shell', 'screencap', '-p', temp_path]
            result1 = subprocess.run(cmd1, capture_output=True, timeout=10)
            
            if result1.returncode != 0:
                logger.error("Failed to take screenshot on device")
                return None
            
            # Pull screenshot from device
            cmd2 = ['adb', '-s', self.device_id, 'pull', temp_path, local_path]
            result2 = subprocess.run(cmd2, capture_output=True, timeout=10)
            
            if result2.returncode != 0:
                logger.error("Failed to pull screenshot from device")
                return None
            
            # Load image with OpenCV
            opencv_image = cv2.imread(local_path)
            
            # Clean up temp files
            subprocess.run(['adb', '-s', self.device_id, 'shell', 'rm', temp_path], 
                         capture_output=True, timeout=5)
            
            import os
            if not save_path and os.path.exists(local_path):
                os.remove(local_path)
            
            return opencv_image
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None
    
    def get_device_info(self) -> dict:
        """Get basic device information (backwards compatibility)"""
        if not self.device_id:
            return {}
        
        detailed_info = self.get_device_detailed_info(self.device_id)
        return {
            'model': detailed_info.get('model', 'Unknown'),
            'android_version': detailed_info.get('android_version', 'Unknown'),
            'resolution': detailed_info.get('resolution', 'Unknown')
        }
    
    def disconnect(self) -> bool:
        """Disconnect from current device"""
        if self.device_id and '127.0.0.1:' in self.device_id:
            try:
                result = subprocess.run(['adb', 'disconnect', self.device_id], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"Disconnected from {self.device_id}")
                    self.device_id = None
                    self.connected = False
                    return True
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
        
        self.device_id = None
        self.connected = False
        return True
    
    def is_package_installed(self, package_name: str, device_id: str = None) -> bool:
        """Check if a package is installed on the device"""
        try:
            target_device = device_id or self.device_id
            if target_device:
                cmd = ['-s', target_device, 'shell', 'pm', 'list', 'packages', package_name]
            else:
                cmd = ['shell', 'pm', 'list', 'packages', package_name]
            
            success, output = self.execute_adb_command(cmd)
            return success and package_name in output
        except Exception as e:
            logger.error(f"Error checking package installation: {e}")
            return False
    
    def get_installed_packages(self, device_id: str = None, filter_packages: list = None) -> list:
        """Get list of specific installed packages on the device"""
        installed_packages = []
        
        if not filter_packages:
            return installed_packages
        
        try:
            target_device = device_id or self.device_id
            
            for package in filter_packages:
                if target_device:
                    cmd = ['-s', target_device, 'shell', 'pm', 'list', 'packages', package]
                else:
                    cmd = ['shell', 'pm', 'list', 'packages', package]
                
                success, output = self.execute_adb_command(cmd)
                if success and package in output:
                    installed_packages.append(package)
                    
        except Exception as e:
            logger.error(f"Error getting installed packages: {e}")
        
        return installed_packages
    
    def tap(self, x: int, y: int) -> bool:
        """
        Tap at specific coordinates on the device screen
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if tap command succeeded, False otherwise
        """
        try:
            cmd = ['shell', 'input', 'tap', str(x), str(y)]
            success, output = self.execute_adb_command(cmd)
            
            if success:
                logger.debug(f"Tapped at ({x}, {y})")
            else:
                logger.warning(f"Failed to tap at ({x}, {y}): {output}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error tapping at ({x}, {y}): {e}")
            return False
    
    def launch_app(self, package_name: str) -> bool:
        """
        Launch an Android app by package name
        
        Args:
            package_name: Package name of the app to launch (e.g., 'com.ncsoft.lineage2m')
            
        Returns:
            True if launch command succeeded, False otherwise
        """
        try:
            # Use 'monkey' command to launch app (works when main activity is unknown)
            # Alternative: Use 'am start' with package name
            cmd = ['shell', 'monkey', '-p', package_name, '-c', 'android.intent.category.LAUNCHER', '1']
            success, output = self.execute_adb_command(cmd)
            
            if success:
                logger.info(f"Launch command sent for {package_name}")
                # Give app time to start
                time.sleep(2)
                return True
            else:
                logger.warning(f"Failed to launch {package_name}: {output}")
                # Try alternative method using 'am start'
                try:
                    cmd = ['shell', 'am', 'start', '-n', f'{package_name}/.MainActivity']
                    success, output = self.execute_adb_command(cmd)
                    if success:
                        logger.info(f"Launched {package_name} using am start")
                        time.sleep(2)
                        return True
                except Exception as e:
                    logger.debug(f"Alternative launch method failed: {e}")
                
                return False
                
        except Exception as e:
            logger.error(f"Error launching app {package_name}: {e}")
            return False
    
    def check_game_status(self, device_id: str = None, game_packages: list = None) -> dict:
        """Check game installation and running status on a device"""
        game_status = {
            'installed_packages': [],
            'running_packages': [],
            'foreground_package': None,
            'game_running': False,
            'game_installed': False
        }
        
        if not game_packages:
            return game_status
        
        try:
            target_device = device_id or self.device_id
            
            if not target_device:
                logger.debug("No target device specified for game status check")
                return game_status
            
            # Check installed packages - execute_adb_command already adds device ID
            for package in game_packages:
                try:
                    cmd = ['shell', 'pm', 'list', 'packages', package]
                    success, output = self.execute_adb_command(cmd)
                    
                    if success and package in output:
                        game_status['installed_packages'].append(package)
                except Exception as e:
                    logger.debug(f"Error checking package {package} on {target_device}: {e}")
                    continue
            
            game_status['game_installed'] = len(game_status['installed_packages']) > 0
            
            # Check running packages - execute_adb_command already adds device ID
            for package in game_status['installed_packages']:
                try:
                    cmd = ['shell', 'pidof', package]
                    success, output = self.execute_adb_command(cmd)
                    
                    if success and output.strip():
                        game_status['running_packages'].append(package)
                except Exception as e:
                    logger.debug(f"Error checking running process {package}: {e}")
                    continue
            
            game_status['game_running'] = len(game_status['running_packages']) > 0
            
            # Check foreground app - execute_adb_command already adds device ID
            try:
                cmd = ['shell', 'dumpsys', 'activity', 'activities']
                success, output = self.execute_adb_command(cmd)
                
                if success and output:
                    lines = output.split('\n')
                    for line in lines:
                        if 'mResumedActivity' in line or 'mCurrentFocus' in line:
                            # Extract package name
                            import re
                            match = re.search(r'([a-zA-Z0-9._]+)/[a-zA-Z0-9._]+', line)
                            if match:
                                foreground_package = match.group(1)
                                if foreground_package in game_packages:
                                    game_status['foreground_package'] = foreground_package
                                break
            except Exception as e:
                logger.debug(f"Error checking foreground app: {e}")
                
        except Exception as e:
            logger.error(f"Error checking game status: {e}")
        
        return game_status
    
    def connect_to_device_with_game(self, device_id: str, game_packages: list) -> bool:
        """Connect to a specific device and verify game is installed"""
        try:
            # Try to connect to the device
            if '127.0.0.1:' in device_id:
                result = subprocess.run(['adb', 'connect', device_id], 
                                      capture_output=True, text=True, timeout=15)
                
                if result.returncode != 0 or 'connected' not in result.stdout.lower():
                    logger.warning(f"Failed to connect to {device_id}")
                    return False
            
            # Check if game is installed on this device
            game_status = self.check_game_status(device_id, game_packages)
            
            if game_status['game_installed']:
                self.device_id = device_id
                self.connected = True
                logger.info(f"Connected to device {device_id} with game installed")
                return True
            else:
                logger.info(f"Device {device_id} connected but no game found")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to device {device_id}: {e}")
            return False
    
    def discover_and_connect_game_devices(self, game_packages: list) -> List[Dict]:
        """Discover and connect to devices that have the game installed"""
        game_ready_devices = []
        
        # Get all potential devices (including BlueStacks)
        all_devices = self.get_all_available_devices()
        
        logger.info(f"Checking {len(all_devices)} devices for game installation...")
        
        for device in all_devices:
            device_id = device['id']
            
            try:
                # Try to connect if not already connected
                if device['status'] != 'connected':
                    logger.debug(f"Attempting to connect to {device_id}")
                    
                    if '127.0.0.1:' in device_id:
                        result = subprocess.run(['adb', 'connect', device_id], 
                                              capture_output=True, text=True, timeout=10)
                        
                        if result.returncode != 0:
                            logger.debug(f"Could not connect to {device_id}")
                            continue
                
                # Check game status
                game_status = self.check_game_status(device_id, game_packages)
                
                if game_status['game_installed']:
                    # Update device info with game status
                    device['game_status'] = game_status
                    device['status'] = 'connected'  # Mark as connected
                    device['game_ready'] = True
                    
                    # Get updated device info now that we're connected
                    detailed_info = self.get_device_detailed_info(device_id)
                    device.update(detailed_info)
                    
                    game_ready_devices.append(device)
                    logger.info(f"Found game-ready device: {device_id}")
                    
                else:
                    logger.debug(f"Device {device_id} does not have game installed")
                    
            except Exception as e:
                logger.debug(f"Error checking device {device_id}: {e}")
                continue
        
        logger.info(f"Found {len(game_ready_devices)} game-ready device(s)")
        return game_ready_devices