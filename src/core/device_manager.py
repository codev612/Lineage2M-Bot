"""
Device Manager - High-level device management and selection
Handles device discovery, selection, and connection management
"""

from typing import List, Dict, Optional, Any
from .adb_manager import ADBManager
from ..utils.logger import get_logger
from ..utils.exceptions import DeviceNotFoundError, ADBError
from ..utils.config import config_manager

logger = get_logger(__name__)

class DeviceManager:
    """
    High-level device management for Android devices and emulators
    Provides device discovery, selection, and connection management
    """
    
    def __init__(self):
        self.adb = ADBManager()
        self.selected_device: Optional[Dict] = None
        self.available_devices: List[Dict] = []
        self.config = config_manager.get_config()
    
    def discover_devices(self) -> List[Dict[str, Any]]:
        """
        Discover all available devices and cache results
        
        Returns:
            List of device information dictionaries with game status
        """
        logger.info("Discovering available devices...")
        
        try:
            # Get all available devices
            self.available_devices = self.adb.get_all_available_devices()
            logger.info(f"Found {len(self.available_devices)} device(s)")
            
            # Enhance device info with game status
            self._enhance_devices_with_game_status()
            
            return self.available_devices
        except Exception as e:
            logger.error(f"Error discovering devices: {e}")
            raise DeviceNotFoundError(f"Failed to discover devices: {e}")
    
    def _enhance_devices_with_game_status(self) -> None:
        """Enhance device information with Lineage 2M game status"""
        game_packages = self.config.game.packages
        
        for device in self.available_devices:
            device_id = device['id']
            logger.debug(f"Checking game status for device: {device_id}")
            
            try:
                # Check game status on device
                game_status = self.adb.check_game_status(device_id, game_packages)
                
                # Add game information to device info
                device['game_status'] = {
                    'installed': game_status['game_installed'],
                    'running': game_status['game_running'],
                    'installed_packages': game_status['installed_packages'],
                    'running_packages': game_status['running_packages'],
                    'foreground_package': game_status['foreground_package']
                }
                
                # Update device status if game is running
                if game_status['game_running']:
                    device['game_active'] = True
                    if game_status['foreground_package']:
                        device['current_game'] = game_status['foreground_package']
                else:
                    device['game_active'] = False
                    
            except Exception as e:
                logger.warning(f"Could not check game status for {device_id}: {e}")
                # Set default game status if check fails
                device['game_status'] = {
                    'installed': False,
                    'running': False,
                    'installed_packages': [],
                    'running_packages': [],
                    'foreground_package': None
                }
                device['game_active'] = False
    
    def discover_game_ready_devices(self) -> List[Dict[str, Any]]:
        """
        Discover and connect to devices that have Lineage 2M installed
        
        Returns:
            List of devices with game installed and connected
        """
        logger.info("Discovering game-ready devices...")
        
        try:
            game_packages = self.config.game.packages
            game_ready_devices = self.adb.discover_and_connect_game_devices(game_packages)
            
            # Add game-ready devices to available devices if not already present  
            for game_device in game_ready_devices:
                # Check if device already exists in available_devices
                existing_device = None
                for existing in self.available_devices:
                    if existing['id'] == game_device['id']:
                        existing_device = existing
                        break
                
                if existing_device:
                    # Update existing device with game-ready status
                    existing_device.update(game_device)
                    existing_device['game_ready'] = True
                else:
                    # Add new game-ready device
                    game_device['game_ready'] = True
                    self.available_devices.append(game_device)
            
            logger.info(f"Added {len(game_ready_devices)} game-ready device(s) to available devices")
            return game_ready_devices
            
        except Exception as e:
            logger.error(f"Error discovering game-ready devices: {e}")
            return []
    
    def discover_devices_with_game_priority(self) -> List[Dict[str, Any]]:
        """
        Discover all devices and prioritize those with game installed
        
        Returns:
            List of all devices with game-ready devices first
        """
        logger.info("Discovering devices with game priority...")
        
        # First discover all devices normally
        all_devices = self.discover_devices()
        
        # Then discover and connect to game-ready devices
        game_ready_devices = self.discover_game_ready_devices()
        
        # Sort devices to prioritize game-ready ones
        game_ready_ids = {device['id'] for device in game_ready_devices}
        
        # Separate game-ready and regular devices
        prioritized_devices = []
        regular_devices = []
        
        for device in self.available_devices:
            if device['id'] in game_ready_ids or device.get('game_ready', False):
                prioritized_devices.append(device)
            else:
                regular_devices.append(device)
        
        # Combine with game-ready devices first
        self.available_devices = prioritized_devices + regular_devices
        
        logger.info(f"Prioritized {len(prioritized_devices)} game-ready device(s)")
        return self.available_devices
    
    def display_devices(self) -> None:
        """Display available devices in a formatted way"""
        if not self.available_devices:
            self.discover_devices()
        
        if not self.available_devices:
            print("❌ No devices found!")
            print("\n💡 Make sure:")
            print("   • BlueStacks or other emulator is running")
            print("   • USB debugging is enabled")
            print("   • Try: adb connect 127.0.0.1:5555")
            return
        
        # Count game-ready devices
        game_ready_count = sum(1 for device in self.available_devices if device.get('game_ready', False))
        
        print(f"\n✅ Found {len(self.available_devices)} device(s):")
        if game_ready_count > 0:
            print(f"🎮 {game_ready_count} device(s) with Lineage 2M installed (listed first)")
        print("-" * 60)
        
        for i, device in enumerate(self.available_devices, 1):
            status_icon = "🟢" if device['status'] == 'connected' else "🟡"
            
            # Game status icons
            game_status = device.get('game_status', {})
            game_icon = ""
            priority_marker = ""
            
            if device.get('game_ready', False):
                priority_marker = " ⭐"  # Star for game-ready devices
                
            if game_status.get('running'):
                game_icon = " 🎮"
            elif game_status.get('installed'):
                game_icon = " 📱"
            
            print(f"{i}. {status_icon} {device['id']}{game_icon}{priority_marker}")
            print(f"   Type: {device['type']}")
            print(f"   Model: {device['model']}")
            print(f"   Android: {device['android_version']} (API {device['api_level']})")
            print(f"   Resolution: {device['resolution']}")
            print(f"   Status: {device['status']}")
            
            # Display game information
            if game_status.get('installed'):
                installed_games = game_status.get('installed_packages', [])
                print(f"   Game: Lineage 2M - {len(installed_games)} package(s) installed")
                
                if game_status.get('running'):
                    running_games = game_status.get('running_packages', [])
                    print(f"   Game Status: RUNNING ({len(running_games)} process(es))")
                    
                    # Show if game is in foreground
                    if game_status.get('foreground_package'):
                        print(f"   Current: {game_status['foreground_package']} (foreground)")
                else:
                    print(f"   Game Status: Installed but not running")
            else:
                print(f"   Game: Lineage 2M not installed")
                
            print("-" * 60)
    
    def select_device_interactive(self) -> bool:
        """
        Interactive device selection
        
        Returns:
            True if device selected successfully, False otherwise
        """
        if not self.available_devices:
            self.discover_devices()
        
        if not self.available_devices:
            return False
        
        # Auto-select if only one device
        if len(self.available_devices) == 1:
            self.selected_device = self.available_devices[0]
            print(f"🎯 Auto-selected: {self.selected_device['id']}")
            return True
        
        # Show devices and let user choose
        self.display_devices()
        
        while True:
            try:
                choice = input(f"\n👆 Select device (1-{len(self.available_devices)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return False
                
                device_index = int(choice) - 1
                if 0 <= device_index < len(self.available_devices):
                    self.selected_device = self.available_devices[device_index]
                    print(f"✅ Selected: {self.selected_device['id']}")
                    return True
                else:
                    print(f"❌ Please enter a number between 1 and {len(self.available_devices)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return False
    
    def select_device_by_id(self, device_id: str) -> bool:
        """
        Select device by ID
        
        Args:
            device_id: Device ID to select
            
        Returns:
            True if device found and selected, False otherwise
        """
        if not self.available_devices:
            self.discover_devices()
        
        for device in self.available_devices:
            if device['id'] == device_id:
                self.selected_device = device
                logger.info(f"Selected device: {device_id}")
                return True
        
        logger.error(f"Device not found: {device_id}")
        return False
    
    def connect_to_selected_device(self) -> bool:
        """
        Connect to the currently selected device
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.selected_device:
            raise ADBError("No device selected")
        
        device_id = self.selected_device['id']
        
        try:
            if self.selected_device['status'] == 'available':
                # Need to connect first
                logger.info(f"Connecting to {device_id}...")
                if not self.adb.connect_to_device(device_id):
                    logger.error(f"Failed to connect to {device_id}")
                    return False
            else:
                # Already connected, just set it
                self.adb.device_id = device_id
                self.adb.connected = True
            
            logger.info(f"Successfully connected to {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to device: {e}")
            return False
    
    def get_selected_device_info(self) -> Dict:
        """Get information about the selected device"""
        if not self.selected_device:
            return {}
        return self.selected_device.copy()
    
    def disconnect(self) -> bool:
        """Disconnect from current device"""
        try:
            result = self.adb.disconnect()
            if result:
                self.selected_device = None
                logger.info("Disconnected from device")
            return result
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to a device"""
        return self.adb.connected and self.adb.device_id is not None
    
    def get_current_device_id(self) -> Optional[str]:
        """Get current device ID"""
        return self.adb.device_id
    
    def refresh_devices(self) -> List[Dict]:
        """Refresh device list and return updated list"""
        logger.info("Refreshing device list...")
        return self.discover_devices()
    
    def get_device_capabilities(self) -> Dict:
        """Get capabilities of the current device"""
        if not self.is_connected():
            return {}
        
        capabilities = {
            'screenshot': True,
            'touch': True,
            'app_detection': True,
            'file_operations': True
        }
        
        # Test screenshot capability
        try:
            screenshot = self.adb.take_screenshot()
            capabilities['screenshot'] = screenshot is not None
        except Exception:
            capabilities['screenshot'] = False
        
        # Test foreground app detection
        try:
            app = self.adb.get_foreground_app()
            capabilities['app_detection'] = app is not None
        except Exception:
            capabilities['app_detection'] = False
        
        return capabilities