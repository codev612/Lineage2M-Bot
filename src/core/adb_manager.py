"""
ADB Manager - Core module for Android Debug Bridge operations
Handles connection to Android devices/emulators and basic ADB operations
"""

import subprocess
import time
import re
import logging
import os
import platform
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
    
    def __init__(self, timeout: int = 30, adb_path: Optional[str] = None):
        """
        Initialize ADB Manager
        
        Args:
            timeout: Default timeout for ADB operations in seconds
            adb_path: Optional path to ADB executable. If None, will auto-detect.
        """
        self.device_id: Optional[str] = None
        self.connected = False
        self.timeout = timeout
        self.adb_path: Optional[str] = adb_path
        self._find_adb()
        self._validate_adb()
    
    def _find_adb(self) -> None:
        """
        Find ADB executable in common locations.
        Search order (priority):
        1. User-provided ADB path (from config)
        2. Standard ADB from system PATH
        3. Standard Android SDK Platform Tools ADB
        4. Device-specific ADB (BlueStacks, MuMu, LDPlayer, Nox) - fallback only
        """
        # Priority 1: If adb_path is provided and exists, use it
        if self.adb_path:
            adb_file = Path(self.adb_path)
            if adb_file.exists() and adb_file.is_file():
                self.adb_path = str(adb_file.absolute())
                logger.info(f"Using provided ADB path: {self.adb_path}")
                return
            else:
                logger.warning(f"Provided ADB path does not exist: {self.adb_path}, will search for ADB")
        
        # Priority 2: Try to find standard ADB in system PATH
        try:
            result = subprocess.run(['adb', 'version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.adb_path = 'adb'  # Use 'adb' from PATH (standard ADB)
                version = result.stdout.strip().split()[0]
                logger.info(f"ADB found in PATH (standard ADB): {version}")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Search common installation locations
        search_paths = []
        
        # Windows paths
        if platform.system() == 'Windows':
            # Android SDK locations
            local_appdata = os.getenv('LOCALAPPDATA', '')
            userprofile = os.getenv('USERPROFILE', '')
            program_files = os.getenv('ProgramFiles', '')
            program_files_x86 = os.getenv('ProgramFiles(x86)', '')
            
            # Priority 3: Standard Android SDK Platform Tools ADB (works with all emulators)
            search_paths.extend([
                Path(local_appdata) / 'Android' / 'Sdk' / 'platform-tools' / 'adb.exe',
                Path(userprofile) / 'AppData' / 'Local' / 'Android' / 'Sdk' / 'platform-tools' / 'adb.exe',
                Path(program_files) / 'Android' / 'android-sdk' / 'platform-tools' / 'adb.exe',
                Path(program_files_x86) / 'Android' / 'android-sdk' / 'platform-tools' / 'adb.exe',
            ])
            
            # Priority 4: Device-specific ADB locations (FALLBACK if standard ADB not found)
            # NOTE: Standard Android SDK ADB works with ALL emulators via network ports.
            # These emulator-specific ADBs are fallbacks for users who don't have
            # Android SDK Platform Tools installed.
            # Best practice: Install Android SDK Platform Tools and use standard ADB.
            search_paths.extend([
                # LDPlayer (LD)
                Path(program_files) / 'LDPlayer' / 'adb.exe',
                Path(program_files_x86) / 'LDPlayer' / 'adb.exe',
                Path(program_files) / 'LDPlayer4.0' / 'adb.exe',
                Path(program_files_x86) / 'LDPlayer4.0' / 'adb.exe',
                # BlueStacks
                Path(program_files) / 'BlueStacks_nxt' / 'HD-Adb.exe',  # BlueStacks 5/nxt
                Path(program_files) / 'BlueStacks' / 'HD-Adb.exe',  # Older BlueStacks
                Path(program_files_x86) / 'BlueStacks_nxt' / 'HD-Adb.exe',
                Path(program_files_x86) / 'BlueStacks' / 'HD-Adb.exe',
                # MuMu Player
                Path(program_files) / 'Netease' / 'MuMuPlayer' / 'nx_device' / '12.0' / 'shell' / 'adb.exe',
                Path(program_files) / 'Netease' / 'MuMuPlayer' / 'shell' / 'adb.exe',
                # Nox
                Path(program_files) / 'Nox' / 'bin' / 'nox_adb.exe',
                Path(program_files_x86) / 'Nox' / 'bin' / 'nox_adb.exe',
            ])
        
        # Linux/Mac paths
        else:
            home = os.getenv('HOME', '')
            search_paths.extend([
                Path(home) / 'Android' / 'Sdk' / 'platform-tools' / 'adb',
                Path(home) / 'Library' / 'Android' / 'sdk' / 'platform-tools' / 'adb',
                Path('/opt' / 'android-sdk' / 'platform-tools' / 'adb'),
                Path('/usr' / 'local' / 'bin' / 'adb'),
            ])
        
        # Search for ADB in common locations (standard ADB first, then device-specific)
        for adb_file in search_paths:
            if adb_file.exists() and adb_file.is_file():
                self.adb_path = str(adb_file.absolute())
                # Determine if it's standard or device-specific ADB
                adb_str = str(adb_file).lower()
                if any(emu in adb_str for emu in ['bluestacks', 'mumu', 'ldplayer', 'nox']):
                    logger.info(f"ADB found at (device-specific fallback): {self.adb_path}")
                else:
                    logger.info(f"ADB found at (standard SDK): {self.adb_path}")
                return
        
        # If still not found, try to search in Program Files recursively (Windows only, slow)
        if platform.system() == 'Windows' and not self.adb_path:
            try:
                import glob
                for drive in ['C:']:
                    pattern = f"{drive}\\Program Files\\**\\adb.exe"
                    matches = glob.glob(pattern, recursive=True)
                    if matches:
                        self.adb_path = matches[0]
                        logger.info(f"ADB found by search: {self.adb_path}")
                        return
            except Exception:
                pass
        
        # If still not found, set to None and let validation handle the error
        if not self.adb_path:
            self.adb_path = None
    
    def _get_adb_command(self, args: List[str]) -> List[str]:
        """Build ADB command with proper path"""
        if self.adb_path:
            return [self.adb_path] + args
        else:
            return ['adb'] + args
    
    def _validate_adb(self) -> None:
        """Validate that ADB is available"""
        if not self.check_adb_available():
            error_msg = "ADB not found. Please install Android SDK Platform Tools or specify ADB path in config."
            if self.adb_path:
                error_msg += f"\nAttempted path: {self.adb_path}"
            raise ADBError(error_msg)
    
    def check_adb_available(self) -> bool:
        """Check if ADB is available"""
        try:
            cmd = self._get_adb_command(['version'])
            result = subprocess.run(cmd, 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip().split()[0]
                logger.info(f"ADB available: {version} at {self.adb_path or 'PATH'}")
                return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"ADB not found: {e}")
            return False
    
    def get_connected_devices(self) -> List[str]:
        """Get list of connected Android devices/emulators"""
        try:
            cmd = self._get_adb_command(['devices'])
            result = subprocess.run(cmd, 
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
        
        # Try to discover emulator instances (BlueStacks, LDPlayer, MuMu, etc.)
        emulator_devices = self.discover_bluestacks_devices()
        for bs_device in emulator_devices:
            # Check if already in connected devices
            if bs_device['id'] not in connected_devices:
                bs_device['status'] = 'available'
                devices.append(bs_device)
        
        return devices
    
    def discover_bluestacks_devices(self) -> List[Dict]:
        """Discover potential emulator instances (BlueStacks, LDPlayer, MuMu, etc.)"""
        emulator_devices = []
        # Common ports for various emulators: BlueStacks, LDPlayer, MuMu, Nox
        # Most emulators use ports starting from 5555, incrementing by 1 or 2
        common_ports = [5555, 5554, 5556, 5558, 5562, 5564, 5566, 5568, 5570, 5572, 5574, 5576, 5578, 5580]
        
        logger.info("Scanning for emulator instances (BlueStacks, LDPlayer, MuMu, etc.)...")
        
        for port in common_ports:
            device_id = f"127.0.0.1:{port}"
            
            # Try to connect temporarily to test
            try:
                cmd = self._get_adb_command(['connect', device_id])
                result = subprocess.run(cmd, 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and 'connected' in result.stdout.lower():
                    # Get device info if connection successful
                    device_info = self.get_device_detailed_info(device_id)
                    device_info['id'] = device_id
                    device_info['port'] = port
                    
                    # Better emulator type detection based on model/manufacturer
                    model = device_info.get('model', '').lower()
                    manufacturer = device_info.get('manufacturer', '').lower()
                    
                    if 'bluestacks' in model or 'bstk' in model or 'bluestacks' in manufacturer:
                        device_info['type'] = 'BlueStacks'
                    elif 'ldplayer' in model or 'ld' in model or 'ldplayer' in manufacturer:
                        device_info['type'] = 'LDPlayer'
                    elif 'mumu' in model or 'mumu' in manufacturer or 'netease' in manufacturer:
                        device_info['type'] = 'MuMu Player'
                    elif 'nox' in model or 'nox' in manufacturer:
                        device_info['type'] = 'Nox Player'
                    else:
                        device_info['type'] = 'Emulator'
                    
                    emulator_devices.append(device_info)
                    
            except (subprocess.TimeoutExpired, Exception) as e:
                # Port not available or no emulator running
                continue
        
        return emulator_devices
    
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
            cmd = self._get_adb_command(['-s', device_id, 'shell', 'getprop', 'ro.product.model'])
            result = subprocess.run(cmd, 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['model'] = result.stdout.strip()
            
            # Get manufacturer
            cmd = self._get_adb_command(['-s', device_id, 'shell', 'getprop', 'ro.product.manufacturer'])
            result = subprocess.run(cmd, 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['manufacturer'] = result.stdout.strip()
            
            # Get Android version
            cmd = self._get_adb_command(['-s', device_id, 'shell', 'getprop', 'ro.build.version.release'])
            result = subprocess.run(cmd, 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['android_version'] = result.stdout.strip()
            
            # Get API level
            cmd = self._get_adb_command(['-s', device_id, 'shell', 'getprop', 'ro.build.version.sdk'])
            result = subprocess.run(cmd, 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['api_level'] = result.stdout.strip()
            
            # Get screen resolution
            cmd = self._get_adb_command(['-s', device_id, 'shell', 'wm', 'size'])
            result = subprocess.run(cmd, 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'Physical size:' in result.stdout:
                resolution = result.stdout.split('Physical size:')[1].strip()
                info['resolution'] = resolution
            
            # Determine device type based on model and manufacturer
            model = info.get('model', '').lower()
            manufacturer = info.get('manufacturer', '').lower()
            
            if '127.0.0.1:' in device_id:
                if 'bluestacks' in model or 'bstk' in model or 'bluestacks' in manufacturer:
                    info['type'] = 'BlueStacks'
                elif 'ldplayer' in model or 'ld' in model or 'ldplayer' in manufacturer:
                    info['type'] = 'LDPlayer'
                elif 'mumu' in model or 'mumu' in manufacturer or 'netease' in manufacturer:
                    info['type'] = 'MuMu Player'
                elif 'nox' in model or 'nox' in manufacturer:
                    info['type'] = 'Nox Player'
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
                
                cmd = self._get_adb_command(['connect', connect_address])
                result = subprocess.run(cmd, 
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
            # For IP addresses (like 127.0.0.1:5555), we need to connect via ADB first
            if ':' in device_id and not device_id.startswith('emulator-'):
                try:
                    logger.info(f"Connecting to device via ADB: {device_id}")
                    cmd = self._get_adb_command(['connect', device_id])
                    result = subprocess.run(cmd, 
                                          capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0 and 'connected' in result.stdout.lower():
                        logger.info(f"Successfully connected to {device_id}")
                    else:
                        # Check if device is already connected
                        connected_devices = self.get_connected_devices()
                        if device_id not in connected_devices:
                            logger.warning(f"Could not connect to {device_id}: {result.stderr or result.stdout}")
                            # Still try to use it, might be connected already
                except Exception as e:
                    logger.warning(f"Error connecting to {device_id}: {e}")
                    # Check if device is already connected anyway
                    connected_devices = self.get_connected_devices()
                    if device_id not in connected_devices:
                        logger.warning(f"Device {device_id} not found in connected devices")
            
            # Verify device is actually available
            connected_devices = self.get_connected_devices()
            if device_id in connected_devices:
                self.device_id = device_id
                self.connected = True
                logger.info(f"Device {device_id} is connected and ready")
                return True
            else:
                logger.warning(f"Device {device_id} not found in connected devices list after connection attempt")
                # Try to discover the correct port
                discovered_device = self._discover_device_port()
                if discovered_device:
                    logger.info(f"Discovered device at {discovered_device}, using that instead")
                    self.device_id = discovered_device
                    self.connected = True
                    return True
                else:
                    logger.error(f"Device {device_id} not available. Please ensure emulator is running and ADB is enabled.")
                    # Don't mark as connected if we can't verify it
                    self.device_id = None
                    self.connected = False
                    return False
    
    def _discover_device_port(self) -> Optional[str]:
        """Discover available emulator device by scanning common ports"""
        common_ports = [5555, 5554, 5556, 5558, 5562, 5564, 5566, 5568, 5570, 5572, 5574, 5576, 5578, 5580]
        
        logger.info("Scanning common ports to discover emulator...")
        for port in common_ports:
            device_id = f"127.0.0.1:{port}"
            try:
                cmd = self._get_adb_command(['connect', device_id])
                result = subprocess.run(cmd, 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and 'connected' in result.stdout.lower():
                    # Verify it's actually connected
                    connected_devices = self.get_connected_devices()
                    if device_id in connected_devices:
                        logger.info(f"Successfully discovered and connected to {device_id}")
                        return device_id
            except Exception:
                continue
        
        return None
    
    def execute_adb_command(self, command: List[str]) -> Tuple[bool, str]:
        """Execute ADB command and return success status and output"""
        if not self.connected or not self.device_id:
            error_msg = f"Not connected - connected={self.connected}, device_id={self.device_id}"
            logger.error(f"ADBManager.execute_adb_command: {error_msg}")
            raise ADBError(error_msg)
        
        full_command = self._get_adb_command(['-s', self.device_id] + command)
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
            cmd1 = self._get_adb_command(['-s', self.device_id, 'shell', 'screencap', '-p', temp_path])
            result1 = subprocess.run(cmd1, capture_output=True, timeout=10)
            
            if result1.returncode != 0:
                logger.error("Failed to take screenshot on device")
                return None
            
            # Pull screenshot from device
            cmd2 = self._get_adb_command(['-s', self.device_id, 'pull', temp_path, local_path])
            result2 = subprocess.run(cmd2, capture_output=True, timeout=10)
            
            if result2.returncode != 0:
                logger.error("Failed to pull screenshot from device")
                return None
            
            # Load image - try OpenCV first, fallback to PIL if OpenCV doesn't have imread
            opencv_image = None
            try:
                if hasattr(cv2, 'imread'):
                    # OpenCV imread loads as BGR by default, but PNG files might have alpha channel
                    # Use IMREAD_UNCHANGED to check, then convert if needed
                    if hasattr(cv2, 'IMREAD_UNCHANGED'):
                        opencv_image = cv2.imread(local_path, cv2.IMREAD_UNCHANGED)
                    else:
                        opencv_image = cv2.imread(local_path)
                    
                    # If image has 4 channels (RGBA), convert to BGR (3 channels)
                    if opencv_image is not None and len(opencv_image.shape) == 3 and opencv_image.shape[2] == 4:
                        logger.debug(f"Image loaded with 4 channels (RGBA), converting to BGR...")
                        # Convert RGBA to BGR (drop alpha channel)
                        if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_RGBA2BGR'):
                            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGR)
                        elif hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGRA2BGR'):
                            # If OpenCV loaded as BGRA, convert BGRA to BGR
                            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGRA2BGR)
                        else:
                            # Manual conversion: drop alpha channel and keep BGR
                            opencv_image = opencv_image[:, :, :3]  # Take only first 3 channels (BGR)
                        logger.debug(f"Converted RGBA to BGR: shape={opencv_image.shape}")
                else:
                    # Fallback: Use PIL to load image and convert to OpenCV format
                    pil_image = Image.open(local_path)
                    # Convert PIL image to numpy array (OpenCV format: BGR)
                    opencv_image = np.array(pil_image)
                    
                    if len(opencv_image.shape) == 3:
                        # Handle RGBA (4 channels) or RGB (3 channels)
                        if opencv_image.shape[2] == 4:
                            # RGBA image - convert to RGB first by removing alpha channel
                            # Or blend alpha channel (preferred for transparency)
                            # For screenshots, we can just drop alpha channel
                            logger.debug(f"Image has 4 channels (RGBA), converting to RGB...")
                            # Option 1: Drop alpha channel (simple)
                            opencv_image = opencv_image[:, :, :3]  # Take only RGB channels
                            # Option 2: Blend alpha if needed (commented out for now)
                            # alpha = opencv_image[:, :, 3:4] / 255.0
                            # rgb = opencv_image[:, :, :3]
                            # opencv_image = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
                        
                        # Convert RGB to BGR for OpenCV compatibility
                        # Use numpy operations instead of cv2.cvtColor if cv2 doesn't have it
                        if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_RGB2BGR'):
                            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                        else:
                            # Manual RGB to BGR conversion using numpy
                            opencv_image = opencv_image[:, :, ::-1]  # Reverse RGB channels to BGR
                    
                    logger.debug(f"Loaded image using PIL fallback: shape={opencv_image.shape}, dtype={opencv_image.dtype}")
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                opencv_image = None
            
            if opencv_image is None:
                logger.error("Failed to load screenshot image")
                # Clean up temp files
                cmd3 = self._get_adb_command(['-s', self.device_id, 'shell', 'rm', temp_path])
                subprocess.run(cmd3, 
                             capture_output=True, timeout=5)
                if not save_path and os.path.exists(local_path):
                    os.remove(local_path)
                return None
            
            # Clean up temp files immediately
            cmd3 = self._get_adb_command(['-s', self.device_id, 'shell', 'rm', temp_path])
            subprocess.run(cmd3, 
                         capture_output=True, timeout=5)
            
            if not save_path and os.path.exists(local_path):
                os.remove(local_path)
            
            # Return screenshot at full device resolution (no downsampling)
            # This ensures all processing uses the same dimensions as the device
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
                cmd = self._get_adb_command(['disconnect', self.device_id])
                result = subprocess.run(cmd, 
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
                cmd = self._get_adb_command(['connect', device_id])
                result = subprocess.run(cmd, 
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
                        cmd = self._get_adb_command(['connect', device_id])
                        result = subprocess.run(cmd, 
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