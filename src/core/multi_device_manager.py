"""
Multi-Device Manager - Manage multiple Android devices simultaneously
Provides parallel device management and control for bot operations
"""

from typing import List, Dict, Set, Optional, Any
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .adb_manager import ADBManager
from .device_manager import DeviceManager
from ..utils.logger import get_logger
from ..utils.exceptions import DeviceNotFoundError, ADBError
from ..utils.config import config_manager

logger = get_logger(__name__)

class DeviceSession:
    """Represents a connected device session with its own ADB manager"""
    
    def __init__(self, device_info: Dict[str, Any]):
        self.device_info = device_info
        self.device_id = device_info['id']
        self.adb = ADBManager()
        self.connected = False
        self.last_activity = time.time()
        self.lock = threading.Lock()
        
    def connect(self) -> bool:
        """Connect to this device"""
        with self.lock:
            try:
                logger.info(f"Connecting to device: {self.device_id}")
                if self.adb.connect_to_device(self.device_id):
                    self.connected = True
                    self.last_activity = time.time()
                    # Verify device_id is actually set in ADBManager
                    if self.adb.device_id == self.device_id:
                        logger.info(f"Successfully connected to {self.device_id} (device_id verified)")
                    else:
                        logger.warning(f"Connected but device_id mismatch! Expected {self.device_id}, got {self.adb.device_id}")
                        # Fix it
                        self.adb.device_id = self.device_id
                        logger.info(f"Fixed device_id to {self.device_id}")
                    return True
                else:
                    logger.error(f"Failed to connect to {self.device_id}")
                    return False
            except Exception as e:
                logger.error(f"Error connecting to {self.device_id}: {e}", exc_info=True)
                return False
    
    def disconnect(self):
        """Disconnect from this device"""
        with self.lock:
            if self.connected:
                logger.info(f"Disconnecting from device: {self.device_id}")
                # ADB doesn't have explicit disconnect, but we mark as disconnected
                self.connected = False
    
    def is_connected(self) -> bool:
        """Check if device is still connected"""
        with self.lock:
            if not self.connected:
                return False
            
            # Check if device is still reachable
            try:
                # ADBManager.execute_adb_command already adds '-s device_id', so don't include it here
                success, _ = self.adb.execute_adb_command(['shell', 'echo', 'test'])
                if success:
                    self.last_activity = time.time()
                    return True
                else:
                    self.connected = False
                    return False
            except Exception:
                self.connected = False
                return False
    
    def execute_command(self, command: List[str]) -> tuple[bool, str]:
        """Execute ADB command on this device"""
        with self.lock:
            if not self.is_connected():
                logger.warning(f"Device {self.device_id} not connected in execute_command")
                return False, "Device not connected"
            
            # Verify ADBManager has device_id set
            if not self.adb.device_id:
                logger.error(f"ADBManager device_id not set for {self.device_id}! Attempting to reconnect...")
                # Try to reconnect
                if self.adb.connect_to_device(self.device_id):
                    logger.info(f"Reconnected ADBManager to {self.device_id}")
                else:
                    logger.error(f"Failed to reconnect ADBManager to {self.device_id}")
                    return False, f"ADBManager device_id not set for {self.device_id}"
            
            try:
                # Log the exact command being executed
                logger.info(f"DeviceSession.execute_command: device_id={self.device_id}, command={command}")
                logger.info(f"ADBManager state: device_id={self.adb.device_id}, connected={self.adb.connected}")
                
                # ADBManager.execute_adb_command already handles device selection internally
                success, output = self.adb.execute_adb_command(command)
                self.last_activity = time.time()
                
                if not success:
                    logger.warning(f"Command failed on {self.device_id}: {output}")
                else:
                    logger.info(f"Command succeeded on {self.device_id}, output length: {len(output) if output else 0}")
                
                return success, output
            except Exception as e:
                logger.error(f"Error executing command on {self.device_id}: {e}", exc_info=True)
                return False, str(e)

class MultiDeviceManager:
    """
    Multi-device manager for simultaneous control of multiple Android devices
    Provides thread-safe operations and parallel command execution
    """
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.connected_devices: Dict[str, DeviceSession] = {}
        self.selected_devices: Set[str] = set()
        self.config = config_manager.get_config()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.Lock()
        
    def discover_devices(self) -> List[Dict[str, Any]]:
        """Discover all available devices"""
        return self.device_manager.discover_devices_with_game_priority()
    
    def connect_device(self, device_id: str, device_info: Dict[str, Any]) -> bool:
        """Connect to a specific device"""
        with self.lock:
            if device_id in self.connected_devices:
                logger.warning(f"Device {device_id} already connected")
                return self.connected_devices[device_id].is_connected()
            
            # Check device limit
            max_devices = config_manager.config.max_devices
            if len(self.connected_devices) >= max_devices:
                logger.warning(f"Cannot connect to {device_id}: Device limit reached ({max_devices})")
                return False
            
            # Create new device session
            session = DeviceSession(device_info)
            if session.connect():
                self.connected_devices[device_id] = session
                logger.info(f"Added device session: {device_id} ({len(self.connected_devices)}/{max_devices} devices)")
                return True
            else:
                return False
    
    def disconnect_device(self, device_id: str) -> bool:
        """Disconnect from a specific device"""
        with self.lock:
            if device_id in self.connected_devices:
                session = self.connected_devices[device_id]
                session.disconnect()
                del self.connected_devices[device_id]
                self.selected_devices.discard(device_id)
                logger.info(f"Disconnected from device: {device_id}")
                return True
            return False
    
    def disconnect_all(self):
        """Disconnect from all devices"""
        with self.lock:
            device_ids = list(self.connected_devices.keys())
            for device_id in device_ids:
                self.disconnect_device(device_id)
            self.selected_devices.clear()
            logger.info("Disconnected from all devices")
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get current device connection status and limits"""
        with self.lock:
            max_devices = config_manager.config.max_devices
            return {
                'connected_count': len(self.connected_devices),
                'max_devices': max_devices,
                'available_slots': max_devices - len(self.connected_devices),
                'connected_devices': list(self.connected_devices.keys()),
                'selected_devices': list(self.selected_devices)
            }
    
    def select_device(self, device_id: str) -> bool:
        """Add device to selection"""
        with self.lock:
            if device_id in self.connected_devices:
                self.selected_devices.add(device_id)
                logger.info(f"Selected device: {device_id}")
                return True
            return False
    
    def deselect_device(self, device_id: str) -> bool:
        """Remove device from selection"""
        with self.lock:
            if device_id in self.selected_devices:
                self.selected_devices.remove(device_id)
                logger.info(f"Deselected device: {device_id}")
                return True
            return False
    
    def toggle_device_selection(self, device_id: str) -> bool:
        """Toggle device selection status"""
        with self.lock:
            if device_id in self.selected_devices:
                return self.deselect_device(device_id)
            else:
                return self.select_device(device_id)
    
    def select_all_connected(self):
        """Select all connected devices"""
        with self.lock:
            self.selected_devices = set(self.connected_devices.keys())
            logger.info(f"Selected all connected devices: {len(self.selected_devices)}")
    
    def deselect_all(self):
        """Deselect all devices"""
        with self.lock:
            self.selected_devices.clear()
            logger.info("Deselected all devices")
    
    def get_connected_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all connected devices"""
        with self.lock:
            result = {}
            for device_id, session in self.connected_devices.items():
                if session.is_connected():
                    result[device_id] = {
                        'device_info': session.device_info,
                        'connected': True,
                        'selected': device_id in self.selected_devices,
                        'last_activity': session.last_activity
                    }
                else:
                    # Remove disconnected devices
                    self.disconnect_device(device_id)
            return result
    
    def get_selected_devices(self) -> List[str]:
        """Get list of selected device IDs"""
        with self.lock:
            # Filter out disconnected devices
            valid_selections = []
            for device_id in self.selected_devices:
                if device_id in self.connected_devices and self.connected_devices[device_id].is_connected():
                    valid_selections.append(device_id)
            self.selected_devices = set(valid_selections)
            return valid_selections
    
    def execute_on_device(self, device_id: str, command, **kwargs):
        """Execute command on a specific device"""
        with self.lock:
            if device_id not in self.connected_devices:
                return None
            
            session = self.connected_devices[device_id]
        
        try:
            if command == 'take_screenshot':
                return session.adb.take_screenshot()
            elif command == 'tap':
                x, y = kwargs.get('x', 0), kwargs.get('y', 0)
                logger.info(f"Executing tap command on {device_id} at coordinates ({x}, {y})")
                success, output = session.execute_command(['shell', 'input', 'tap', str(x), str(y)])
                if not success:
                    logger.error(f"Tap command failed on {device_id}: {output}")
                else:
                    logger.info(f"Tap command executed successfully on {device_id} (output: {output})")
                return success
            elif command == 'swipe':
                x1, y1 = kwargs.get('x1', 0), kwargs.get('y1', 0)
                x2, y2 = kwargs.get('x2', 0), kwargs.get('y2', 0)
                logger.info(f"Executing swipe command on {device_id} from ({x1}, {y1}) to ({x2}, {y2})")
                # Add duration parameter for swipe (in milliseconds, default 300ms)
                duration = kwargs.get('duration', 300)
                success, output = session.execute_command(['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(duration)])
                if not success:
                    logger.error(f"Swipe command failed on {device_id}: {output}")
                else:
                    logger.info(f"Swipe command executed successfully on {device_id} (output: {output})")
                return success
            elif isinstance(command, list):
                # Legacy command list support
                success, output = session.execute_command(command)
                return success, output
            else:
                logger.warning(f"Unknown command type: {command}")
                return None
        except Exception as e:
            logger.error(f"Error executing command {command} on {device_id}: {e}")
            return None
    
    def execute_on_selected(self, command: List[str]) -> Dict[str, tuple[bool, str]]:
        """Execute command on all selected devices in parallel"""
        selected_devices = self.get_selected_devices()
        if not selected_devices:
            return {}
        
        logger.info(f"Executing command on {len(selected_devices)} selected devices")
        
        # Execute commands in parallel
        future_to_device = {}
        results = {}
        
        for device_id in selected_devices:
            future = self.executor.submit(self.execute_on_device, device_id, command)
            future_to_device[future] = device_id
        
        # Collect results
        for future in as_completed(future_to_device):
            device_id = future_to_device[future]
            try:
                success, output = future.result(timeout=30)  # 30 second timeout
                results[device_id] = (success, output)
            except Exception as e:
                logger.error(f"Error executing command on {device_id}: {e}")
                results[device_id] = (False, str(e))
        
        return results
    
    def execute_on_all(self, command: List[str]) -> Dict[str, tuple[bool, str]]:
        """Execute command on all connected devices in parallel"""
        with self.lock:
            device_ids = list(self.connected_devices.keys())
        
        if not device_ids:
            return {}
        
        logger.info(f"Executing command on {len(device_ids)} connected devices")
        
        # Execute commands in parallel
        future_to_device = {}
        results = {}
        
        for device_id in device_ids:
            future = self.executor.submit(self.execute_on_device, device_id, command)
            future_to_device[future] = device_id
        
        # Collect results
        for future in as_completed(future_to_device):
            device_id = future_to_device[future]
            try:
                success, output = future.result(timeout=30)
                results[device_id] = (success, output)
            except Exception as e:
                logger.error(f"Error executing command on {device_id}: {e}")
                results[device_id] = (False, str(e))
        
        return results
    
    def take_screenshots_parallel(self) -> Dict[str, Optional[str]]:
        """Take screenshots on all selected devices in parallel"""
        selected_devices = self.get_selected_devices()
        if not selected_devices:
            return {}
        
        logger.info(f"Taking screenshots on {len(selected_devices)} devices")
        
        def take_screenshot(device_id: str) -> Optional[str]:
            try:
                with self.lock:
                    if device_id not in self.connected_devices:
                        return None
                    session = self.connected_devices[device_id]
                
                # Take screenshot using the device's ADB manager
                screenshot_path = session.adb.take_screenshot(save_path=f"screenshots/multi_{device_id}_{int(time.time())}.png")
                return screenshot_path
            except Exception as e:
                logger.error(f"Error taking screenshot on {device_id}: {e}")
                return None
        
        # Execute in parallel
        future_to_device = {}
        results = {}
        
        for device_id in selected_devices:
            future = self.executor.submit(take_screenshot, device_id)
            future_to_device[future] = device_id
        
        # Collect results
        for future in as_completed(future_to_device):
            device_id = future_to_device[future]
            try:
                screenshot_path = future.result(timeout=15)
                results[device_id] = screenshot_path
            except Exception as e:
                logger.error(f"Error taking screenshot on {device_id}: {e}")
                results[device_id] = None
        
        return results
    
    def get_game_status_all(self) -> Dict[str, Dict[str, Any]]:
        """Get game status for all connected devices"""
        with self.lock:
            device_ids = list(self.connected_devices.keys())
        
        if not device_ids:
            return {}
        
        logger.info(f"Checking game status on {len(device_ids)} devices")
        
        def get_game_status(device_id: str) -> Dict[str, Any]:
            try:
                with self.lock:
                    if device_id not in self.connected_devices:
                        return {'error': 'Device not connected'}
                    session = self.connected_devices[device_id]
                
                # Get game packages from config
                game_packages = self.config.game.packages
                
                # Check game status using the device's ADB manager
                status = session.adb.check_game_status(device_id, game_packages)
                return status
            except Exception as e:
                logger.error(f"Error checking game status on {device_id}: {e}")
                return {'error': str(e)}
        
        # Execute in parallel
        future_to_device = {}
        results = {}
        
        for device_id in device_ids:
            future = self.executor.submit(get_game_status, device_id)
            future_to_device[future] = device_id
        
        # Collect results
        for future in as_completed(future_to_device):
            device_id = future_to_device[future]
            try:
                status = future.result(timeout=15)
                results[device_id] = status
            except Exception as e:
                logger.error(f"Error getting game status for {device_id}: {e}")
                results[device_id] = {'error': str(e)}
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self.disconnect_all()
        self.executor.shutdown(wait=True)
        logger.info("Multi-device manager cleaned up")
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass