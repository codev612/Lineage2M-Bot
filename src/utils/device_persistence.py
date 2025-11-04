"""
Device Persistence Manager
Handles saving and loading manually added devices
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

DEVICES_FILE = Path("config/saved_devices.json")


class DevicePersistence:
    """Manages persistence of manually added devices"""
    
    def __init__(self, devices_file: Path = DEVICES_FILE):
        """
        Initialize device persistence manager
        
        Args:
            devices_file: Path to the JSON file storing device IDs
        """
        self.devices_file = devices_file
        self.devices_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_device(self, device_id: str) -> bool:
        """
        Save a device ID to the persistent storage
        
        Args:
            device_id: Device ID to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            devices = self.load_devices()
            
            # Add device if not already present
            if device_id not in devices:
                devices.append(device_id)
                logger.info(f"Saving device: {device_id}")
            else:
                logger.debug(f"Device {device_id} already in saved list")
            
            # Save to file
            with open(self.devices_file, 'w', encoding='utf-8') as f:
                json.dump(devices, f, indent=2)
            
            logger.info(f"Saved {len(devices)} device(s) to {self.devices_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving device {device_id}: {e}")
            return False
    
    def load_devices(self) -> List[str]:
        """
        Load saved device IDs from persistent storage
        
        Returns:
            List of device IDs
        """
        try:
            if self.devices_file.exists():
                with open(self.devices_file, 'r', encoding='utf-8') as f:
                    devices = json.load(f)
                    if isinstance(devices, list):
                        logger.info(f"Loaded {len(devices)} saved device(s) from {self.devices_file}")
                        return devices
                    else:
                        logger.warning("Invalid devices file format, returning empty list")
                        return []
            else:
                logger.info("No saved devices file found, returning empty list")
                return []
        except Exception as e:
            logger.error(f"Error loading devices: {e}")
            return []
    
    def remove_device(self, device_id: str) -> bool:
        """
        Remove a device ID from persistent storage
        
        Args:
            device_id: Device ID to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            devices = self.load_devices()
            
            if device_id in devices:
                devices.remove(device_id)
                
                # Save updated list
                with open(self.devices_file, 'w', encoding='utf-8') as f:
                    json.dump(devices, f, indent=2)
                
                logger.info(f"Removed device: {device_id}")
                return True
            else:
                logger.debug(f"Device {device_id} not in saved list")
                return False
                
        except Exception as e:
            logger.error(f"Error removing device {device_id}: {e}")
            return False
    
    def clear_devices(self) -> bool:
        """
        Clear all saved devices
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            with open(self.devices_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            
            logger.info("Cleared all saved devices")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing devices: {e}")
            return False


# Global device persistence instance
device_persistence = DevicePersistence()

