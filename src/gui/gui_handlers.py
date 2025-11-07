"""
GUI Event Handlers - Methods for handling GUI events and user interactions
Extension of the MainWindow class with all event handling methods
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
from typing import List, Dict, Any, Optional
import threading
import time
import json
import yaml
from pathlib import Path
import subprocess
import webbrowser
from PIL import Image, ImageTk
import cv2
import numpy as np

from ..core.device_manager import DeviceManager
from ..core.multi_device_manager import MultiDeviceManager
from ..modules.game_detector import GameDetector
from ..modules.game_automation import GameAutomation
from ..utils.config import config_manager
from ..utils.logger import get_logger
from ..utils.exceptions import BotError
from ..utils.device_persistence import device_persistence
from ..utils.device_state_monitor import device_state_monitor

logger = get_logger(__name__)

class GUIEventHandlers:
    """
    Event handlers and methods for the main GUI window
    This class contains all the event handling methods for MainWindow
    """
    
    def _discover_devices(self):
        """Discover available devices and auto-connect to game-ready devices"""
        def discover_thread():
            try:
                self._update_status("🔍 Discovering devices...")
                self._show_progress()
                
                # Discover devices with game priority
                devices = self.device_manager.discover_devices_with_game_priority()
                self.devices_list = devices
                
                # Auto-connect to game-ready devices
                game_ready_devices = []
                for device in devices:
                    game_status = device.get('game_status', {})
                    if game_status.get('installed') or game_status.get('running'):
                        game_ready_devices.append(device)
                
                # Connect to game-ready devices automatically
                if game_ready_devices:
                    device_status = self.multi_device_manager.get_device_status()
                    max_devices = device_status['max_devices']
                    available_slots = device_status['available_slots']
                    
                    devices_to_connect = min(len(game_ready_devices), available_slots)
                    self._update_status(f"🎮 Auto-connecting to {devices_to_connect}/{len(game_ready_devices)} game-ready devices (Limit: {max_devices})...")
                    
                    connected_count = 0
                    for device in game_ready_devices[:devices_to_connect]:  # Only try to connect up to available slots
                        try:
                            actual_device_id = self.multi_device_manager.connect_device(device['id'], device)
                            if actual_device_id:
                                connected_count += 1
                                logger.info(f"Auto-connected to game-ready device: {actual_device_id}")
                        except Exception as e:
                            logger.error(f"Failed to auto-connect to {device['id']}: {e}")
                    
                    # Get updated device status
                    final_status = self.multi_device_manager.get_device_status()
                    
                    if connected_count > 0:
                        self.message_queue.put({
                            'type': 'devices_discovered_and_connected',
                            'devices': devices,
                            'connected_count': connected_count,
                            'device_status': final_status
                        })
                    else:
                        self.message_queue.put({
                            'type': 'devices_discovered',
                            'devices': devices,
                            'device_status': final_status
                        })
                else:
                    # No game-ready devices found
                    device_status = self.multi_device_manager.get_device_status()
                    self.message_queue.put({
                        'type': 'devices_discovered',
                        'devices': devices,
                        'device_status': device_status
                    })
                
            except Exception as e:
                logger.error(f"Error discovering devices: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Device discovery failed: {e}"
                })
            finally:
                self._hide_progress()
        
        # Run in separate thread
        threading.Thread(target=discover_thread, daemon=True).start()
    
    def _refresh_devices(self):
        """Refresh the device list"""
        self._discover_devices()
    
    def _restore_saved_devices(self):
        """Restore saved devices on startup - gracefully handles connection failures"""
        # First, load and show saved devices immediately (before connection attempts)
        try:
            saved_devices = device_persistence.load_devices()
            
            if saved_devices:
                logger.info(f"Loading {len(saved_devices)} saved device(s) to display...")
                # Initialize devices_list if needed
                if not hasattr(self, 'devices_list') or self.devices_list is None:
                    self.devices_list = []
                
                # Add all saved devices immediately with "connecting" status
                for device_id in saved_devices:
                    existing_ids = [d['id'] for d in self.devices_list]
                    if device_id not in existing_ids:
                        pending_device_info = {
                            'id': device_id,
                            'type': 'Emulator' if ':' in device_id else 'Unknown',
                            'model': 'Unknown',
                            'android_version': 'Unknown',
                            'resolution': 'Unknown',
                            'status': 'pending',
                            'connection_error': None,
                            'game_status': {
                                'installed': False,
                                'running': False,
                                'installed_packages': [],
                                'running_packages': [],
                                'foreground_package': None
                            },
                            'is_saved_device': True
                        }
                        self.devices_list.append(pending_device_info)
                
                # Update GUI immediately to show saved devices
                if self.devices_list:
                    self.message_queue.put({
                        'type': 'update_device_list',
                        'devices': self.devices_list
                    })
                    logger.info(f"Showing {len(saved_devices)} saved device(s) in GUI")
        except Exception as e:
            logger.error(f"Error loading saved devices for display: {e}", exc_info=True)
        
        # Now try to connect in background thread
        def restore_thread():
            try:
                saved_devices = device_persistence.load_devices()
                
                if not saved_devices:
                    logger.info("No saved devices to restore")
                    return
                
                logger.info(f"Attempting to connect to {len(saved_devices)} saved device(s)...")
                self._update_status(f"🔄 Connecting to {len(saved_devices)} saved device(s)...")
                
                restored_count = 0
                failed_devices = []
                
                for device_id in saved_devices:
                    try:
                        # Try to connect via ADB if it's an IP address
                        if ':' in device_id and not device_id.startswith('emulator-'):
                            try:
                                subprocess.run(
                                    ['adb', 'connect', device_id],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                            except Exception as e:
                                logger.debug(f"ADB connect attempt failed for {device_id}: {e}")
                        
                        # Ensure ADB is connected to this device
                        connection_error = None
                        try:
                            if not self.device_manager.adb.connect_to_device(device_id):
                                connection_error = "Device offline or using different port"
                                logger.warning(f"Could not connect to saved device: {device_id} - {connection_error}")
                                failed_devices.append((device_id, connection_error))
                                # Update existing device in list with error status
                                if hasattr(self, 'devices_list') and self.devices_list:
                                    for device in self.devices_list:
                                        if device['id'] == device_id:
                                            device['status'] = 'disconnected'
                                            device['connection_error'] = connection_error
                                            break
                                # Update connection state to false
                                try:
                                    device_state_monitor.update_connection_state(device_id, False)
                                except Exception:
                                    pass
                                continue
                        except Exception as e:
                            connection_error = f"Connection error: {str(e)}"
                            logger.error(f"Exception while connecting to {device_id}: {e}", exc_info=True)
                            failed_devices.append((device_id, connection_error))
                            # Update existing device in list with error status
                            if hasattr(self, 'devices_list') and self.devices_list:
                                for device in self.devices_list:
                                    if device['id'] == device_id:
                                        device['status'] = 'disconnected'
                                        device['connection_error'] = connection_error
                                        break
                            continue
                        
                        # Get device information
                        device_info = self.device_manager.adb.get_device_detailed_info(device_id)
                        device_info['id'] = device_id
                        device_info['status'] = 'connected'
                        
                        # Determine device type
                        if device_id.startswith('127.0.0.1:') or device_id.startswith('emulator-'):
                            device_info['type'] = 'Emulator'
                        elif 'BlueStacks' in device_info.get('model', ''):
                            device_info['type'] = 'BlueStacks'
                        else:
                            device_info['type'] = 'Physical Device'
                        
                        # Check game status
                        game_packages = self.config.game.packages
                        try:
                            game_status = self.device_manager.adb.check_game_status(device_id, game_packages)
                            device_info['game_status'] = {
                                'installed': game_status.get('game_installed', False),
                                'running': game_status.get('game_running', False),
                                'installed_packages': game_status.get('installed_packages', []),
                                'running_packages': game_status.get('running_packages', []),
                                'foreground_package': game_status.get('foreground_package', None)
                            }
                            
                            if device_info['game_status']['running']:
                                device_info['game_active'] = True
                                if game_status.get('foreground_package'):
                                    device_info['current_game'] = game_status['foreground_package']
                            else:
                                device_info['game_active'] = False
                        except Exception as e:
                            logger.warning(f"Could not check game status for {device_id}: {e}")
                            device_info['game_status'] = {
                                'installed': False,
                                'running': False,
                                'installed_packages': [],
                                'running_packages': [],
                                'foreground_package': None
                            }
                            device_info['game_active'] = False
                        
                        # Connect to device using multi-device manager
                        actual_device_id = self.multi_device_manager.connect_device(device_id, device_info)
                        if actual_device_id:
                            restored_count += 1
                            # Device may have been discovered at a different port
                            if actual_device_id != device_id:
                                logger.info(f"Device port changed during restore: {device_id} -> {actual_device_id}")
                                # Update device_info with the actual device_id
                                device_info['id'] = actual_device_id
                                # Update saved device list to use the discovered port
                                try:
                                    device_persistence.remove_device(device_id)
                                    device_persistence.save_device(actual_device_id)
                                    logger.info(f"Updated saved device from {device_id} to {actual_device_id}")
                                except Exception as e:
                                    logger.warning(f"Could not update saved device list: {e}")
                            
                            logger.info(f"Restored device: {actual_device_id}")
                            
                            # Register device with state monitor (use actual_device_id)
                            device_state_monitor.register_device(actual_device_id)
                            device_state_monitor.update_connection_state(actual_device_id, True)
                            
                            # Update game state in monitor if game is running
                            if device_info.get('game_status', {}).get('running'):
                                running_package = device_info['game_status'].get('foreground_package') or \
                                                 device_info['game_status'].get('running_packages', [None])[0]
                                device_state_monitor.update_game_state(
                                    actual_device_id,
                                    is_running=True,
                                    package_name=running_package,
                                    game_state='detected'
                                )
                                logger.info(f"Updated game state for restored {actual_device_id}: running={True}, package={running_package}")
                            else:
                                device_state_monitor.update_game_state(actual_device_id, is_running=False)
                                logger.info(f"Updated game state for restored {actual_device_id}: running=False")
                            
                            # Update existing device in list with connected info
                            if hasattr(self, 'devices_list') and self.devices_list:
                                # Remove old device_id if it changed
                                if actual_device_id != device_id:
                                    self.devices_list = [d for d in self.devices_list if d['id'] != device_id]
                                
                                # Update or add device with actual_device_id
                                found = False
                                for i, device in enumerate(self.devices_list):
                                    if device['id'] == actual_device_id:
                                        # Update with connected device info
                                        self.devices_list[i] = device_info
                                        found = True
                                        break
                                if not found:
                                    self.devices_list.append(device_info)
                            else:
                                # Fallback: add to list if not found
                                if not hasattr(self, 'devices_list') or self.devices_list is None:
                                    self.devices_list = []
                                self.devices_list.append(device_info)
                        else:
                            logger.warning(f"Failed to connect to saved device: {device_id}")
                            device_state_monitor.update_connection_state(device_id, False)
                            
                    except Exception as e:
                        connection_error = f"Error: {str(e)}"
                        logger.error(f"Error restoring device {device_id}: {e}", exc_info=True)
                        failed_devices.append((device_id, connection_error))
                        # Update existing device in list with error status
                        if hasattr(self, 'devices_list') and self.devices_list:
                            for device in self.devices_list:
                                if device['id'] == device_id:
                                    device['status'] = 'disconnected'
                                    device['connection_error'] = connection_error
                                    break
                        # Ensure connection state is marked as false
                        try:
                            device_state_monitor.update_connection_state(device_id, False)
                        except Exception:
                            pass
                        continue
                
                # Update GUI with restored devices
                try:
                    if restored_count > 0:
                        device_status = self.multi_device_manager.get_device_status()
                        self.message_queue.put({
                            'type': 'devices_restored',
                            'devices': self.devices_list,
                            'restored_count': restored_count,
                            'device_status': device_status
                        })
                    else:
                        # No devices restored, but don't break the GUI
                        logger.info("No devices were successfully restored, but GUI will continue to work")
                        self.message_queue.put({
                            'type': 'devices_restore_failed',
                            'message': 'Could not restore any saved devices'
                        })
                    
                    # Update GUI with all devices (including failed ones) after connection attempts
                    if self.devices_list:
                        self.message_queue.put({
                            'type': 'update_device_list',
                            'devices': self.devices_list
                        })
                        logger.info(f"Updated GUI with {len(self.devices_list)} device(s) (connected: {restored_count}, failed: {len(failed_devices)})")
                    
                    # Log summary
                    if failed_devices:
                        failed_list = [f"{dev_id} ({error})" for dev_id, error in failed_devices]
                        logger.warning(f"Failed to restore {len(failed_devices)} device(s): {failed_list}")
                        logger.info("💡 Tip: Failed devices are shown in the device list with error messages")
                    
                    if restored_count > 0:
                        self._update_status(f"✅ Restored {restored_count} device(s)" + (f", {len(failed_devices)} failed" if failed_devices else ""))
                    elif failed_devices:
                        self._update_status(f"⚠️ Could not connect to {len(failed_devices)} saved device(s) - shown in device list")
                    else:
                        self._update_status("⚠️ No saved devices to restore")
                        
                except Exception as e:
                    logger.error(f"Error updating GUI with restored devices: {e}", exc_info=True)
                    # Even if GUI update fails, don't break - just log the error
                    
            except Exception as e:
                logger.error(f"Error restoring saved devices: {e}", exc_info=True)
                # Don't break the GUI - just log the error and continue
                try:
                    self.message_queue.put({
                        'type': 'error',
                        'message': f"Error restoring saved devices: {e}"
                    })
                except Exception:
                    pass  # If even message queue fails, just continue
                logger.info("GUI will continue to work - you can manually discover devices")
        
        # Run in separate thread to avoid blocking GUI startup
        threading.Thread(target=restore_thread, daemon=True).start()
    
    def _add_device_manually(self):
        """Add device manually by entering device ID"""
        # Show input dialog
        dialog = ctk.CTkInputDialog(
            text="Enter device ID (e.g., 127.0.0.1:5555 or emulator-5554):",
            title="Add Device Manually"
        )
        device_id = dialog.get_input()
        
        if not device_id or not device_id.strip():
            return
        
        device_id = device_id.strip()
        
        def connect_thread():
            try:
                self._update_status(f"🔗 Connecting to {device_id}...")
                self._show_progress()
                
                # First, try to connect via ADB if it's an IP address
                if ':' in device_id and not device_id.startswith('emulator-'):
                    try:
                        result = subprocess.run(
                            ['adb', 'connect', device_id],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode != 0:
                            logger.warning(f"ADB connect returned non-zero: {result.stderr}")
                    except Exception as e:
                        logger.warning(f"Could not connect via ADB connect: {e}")
                
                # Ensure ADB is connected to this device before getting info
                if not self.device_manager.adb.connect_to_device(device_id):
                    error_msg = f"Failed to establish ADB connection to {device_id}"
                    logger.error(error_msg)
                    self.message_queue.put({
                        'type': 'error',
                        'message': error_msg
                    })
                    return
                
                # Get device information
                device_info = self.device_manager.adb.get_device_detailed_info(device_id)
                device_info['id'] = device_id
                device_info['status'] = 'connected'
                
                # Determine device type
                if device_id.startswith('127.0.0.1:') or device_id.startswith('emulator-'):
                    device_info['type'] = 'Emulator'
                elif 'BlueStacks' in device_info.get('model', ''):
                    device_info['type'] = 'BlueStacks'
                else:
                    device_info['type'] = 'Physical Device'
                
                # Check game status (device must be connected first)
                game_packages = self.config.game.packages
                try:
                    # Ensure device_id is set in ADB manager for check_game_status
                    logger.info(f"Checking game status for device {device_id}...")
                    game_status = self.device_manager.adb.check_game_status(device_id, game_packages)
                    logger.info(f"Game status result: installed={game_status.get('game_installed')}, running={game_status.get('game_running')}, running_packages={game_status.get('running_packages')}")
                    
                    device_info['game_status'] = {
                        'installed': game_status.get('game_installed', False),
                        'running': game_status.get('game_running', False),
                        'installed_packages': game_status.get('installed_packages', []),
                        'running_packages': game_status.get('running_packages', []),
                        'foreground_package': game_status.get('foreground_package', None)
                    }
                    
                    # Set game_active flag if game is running
                    if device_info['game_status']['running']:
                        device_info['game_active'] = True
                        if game_status.get('foreground_package'):
                            device_info['current_game'] = game_status['foreground_package']
                    else:
                        device_info['game_active'] = False
                        
                except Exception as e:
                    logger.error(f"Could not check game status: {e}", exc_info=True)
                    device_info['game_status'] = {
                        'installed': False,
                        'running': False,
                        'installed_packages': [],
                        'running_packages': [],
                        'foreground_package': None
                    }
                    device_info['game_active'] = False
                
                # Connect to device using multi-device manager
                actual_device_id = self.multi_device_manager.connect_device(device_id, device_info)
                if actual_device_id:
                    # Device may have been discovered at a different port
                    if actual_device_id != device_id:
                        logger.info(f"Device port changed during connection: {device_id} -> {actual_device_id}")
                        device_info['id'] = actual_device_id
                    
                    logger.info(f"Successfully connected to manually added device: {actual_device_id}")
                    
                    # Register device with state monitor (use actual_device_id)
                    device_state_monitor.register_device(actual_device_id)
                    device_state_monitor.update_connection_state(actual_device_id, True)
                    
                    # Update game state in monitor if game is running
                    if device_info.get('game_status', {}).get('running'):
                        running_package = device_info['game_status'].get('foreground_package') or \
                                         device_info['game_status'].get('running_packages', [None])[0]
                        device_state_monitor.update_game_state(
                            actual_device_id,
                            is_running=True,
                            package_name=running_package,
                            game_state='detected'
                        )
                        logger.info(f"Updated game state for {actual_device_id}: running={True}, package={running_package}")
                    else:
                        device_state_monitor.update_game_state(actual_device_id, is_running=False)
                        logger.info(f"Updated game state for {actual_device_id}: running=False")
                    
                    # Save device to persistent storage (use actual_device_id)
                    device_persistence.save_device(actual_device_id)
                    
                    # Add device to existing list without doing full discovery
                    if not hasattr(self, 'devices_list') or self.devices_list is None:
                        self.devices_list = []
                    
                    # Remove old device_id from list if port changed
                    if actual_device_id != device_id:
                        self.devices_list = [d for d in self.devices_list if d['id'] != device_id]
                    
                    # Check if device already in list (use actual_device_id)
                    existing_ids = [d['id'] for d in self.devices_list]
                    if actual_device_id not in existing_ids:
                        self.devices_list.append(device_info)
                    
                    device_status = self.multi_device_manager.get_device_status()
                    self.message_queue.put({
                        'type': 'device_added_manually',
                        'device_id': actual_device_id,  # Use actual device_id
                        'devices': self.devices_list,
                        'device_status': device_status
                    })
                else:
                    error_msg = f"Failed to connect to {device_id}. Make sure the device is accessible and ADB is working."
                    logger.error(error_msg)
                    self.message_queue.put({
                        'type': 'error',
                        'message': error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Error adding device {device_id}: {e}"
                logger.error(error_msg, exc_info=True)
                self.message_queue.put({
                    'type': 'error',
                    'message': error_msg
                })
            finally:
                self._hide_progress()
        
        # Run in separate thread
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def _connect_to_device(self):
        """Connect to the selected device (legacy single-device mode)"""
        if not self.selected_device:
            messagebox.showwarning("No Device", "Please select a device first.")
            return
        
        def connect_thread():
            try:
                self._update_status("🔗 Connecting to device...")
                self._show_progress()
                
                # Connect to device
                if self.device_manager.select_device_by_id(self.selected_device['id']):
                    if self.device_manager.connect_to_selected_device():
                        # Initialize game detector
                        self.game_detector = GameDetector(
                            self.device_manager.adb, 
                            self.config.game
                        )
                        
                        self.message_queue.put({
                            'type': 'device_connected',
                            'device': self.selected_device
                        })
                    else:
                        raise BotError("Failed to connect to device")
                else:
                    raise BotError("Failed to select device")
                    
            except Exception as e:
                logger.error(f"Error connecting to device: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Connection failed: {e}"
                })
            finally:
                self._hide_progress()
        
        # Run in separate thread
        threading.Thread(target=connect_thread, daemon=True).start()
    

    
    def _select_all_devices(self):
        """Select all available devices"""
        if not hasattr(self, 'devices_list'):
            messagebox.showwarning("No Devices", "Please discover devices first.")
            return
        
        # Toggle all devices in the tree
        for item in self.device_tree.get_children():
            values = list(self.device_tree.item(item, "values"))
            values[0] = "☑️"  # Select checkbox
            self.device_tree.item(item, values=values)
        
        self._update_button_states()
    
    def _disconnect_all_devices(self):
        """Disconnect from all devices"""
        def disconnect_thread():
            try:
                self._update_status("🚫 Disconnecting from all devices...")
                self._show_progress()
                
                # Get all connected devices before disconnecting
                connected_devices = list(self.multi_device_manager.get_connected_devices().keys())
                
                self.multi_device_manager.disconnect_all()
                
                # Update state monitor for all disconnected devices
                for device_id in connected_devices:
                    device_state_monitor.update_connection_state(device_id, False)
                    device_state_monitor.update_bot_state(device_id, False, "Disconnected")
                
                self.message_queue.put({
                    'type': 'all_devices_disconnected'
                })
                
            except Exception as e:
                logger.error(f"Error disconnecting from devices: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Failed to disconnect from devices: {e}"
                })
            finally:
                self._hide_progress()
        
        # Run in separate thread
        threading.Thread(target=disconnect_thread, daemon=True).start()
    
    def _get_selected_devices_from_tree(self) -> List[Dict[str, Any]]:
        """Get devices that are selected in the tree view"""
        selected_devices = []
        seen_device_ids = set()  # Track seen device IDs to prevent duplicates
        
        if not hasattr(self, 'devices_list'):
            return selected_devices
        
        for item in self.device_tree.get_children():
            values = self.device_tree.item(item, "values")
            if len(values) > 0 and values[0] == "☑️":  # Checkbox is checked
                device_id = self.device_tree.item(item, "text").replace(" ⭐", "")  # Remove star
                
                # Skip if we've already seen this device_id
                if device_id in seen_device_ids:
                    logger.warning(f"Duplicate device_id found in tree: {device_id}, skipping")
                    continue
                
                seen_device_ids.add(device_id)
                
                # Find the corresponding device info
                for device in self.devices_list:
                    if device['id'] == device_id:
                        selected_devices.append(device)
                        break
        
        return selected_devices
    
    def _auto_select_connected_devices(self, connected_devices: Dict[str, Any]):
        """Auto-select (check) connected devices in the device tree"""
        if not hasattr(self, 'device_tree'):
            return
        
        # Update each item in the tree
        for item in self.device_tree.get_children():
            device_id = self.device_tree.item(item, "text").replace(" ⭐", "")  # Remove star
            values = list(self.device_tree.item(item, "values"))
            
            if len(values) > 0:
                # Auto-select connected devices
                if device_id in connected_devices:
                    values[0] = "☑️"  # Check the checkbox
                    logger.info(f"Auto-selected connected device: {device_id}")
                else:
                    values[0] = "☐"   # Uncheck if not connected
                
                self.device_tree.item(item, values=values)
    
    def _refresh_device_connections(self):
        """Refresh the connection status display in the device tree"""
        connected_devices = self.multi_device_manager.get_connected_devices()
        
        # Update each item in the tree
        for item in self.device_tree.get_children():
            device_id = self.device_tree.item(item, "text").replace(" ⭐", "")  # Remove star
            values = list(self.device_tree.item(item, "values"))
            
            if len(values) >= 8:  # Ensure we have enough columns
                # Update connection status column
                if device_id in connected_devices:
                    values[7] = "🟢 Connected"  # Connection column
                else:
                    values[7] = "🔴 Disconnected"
                
                self.device_tree.item(item, values=values)
    
    def _on_device_select(self, event):
        """Handle device selection in the tree - toggle checkbox"""
        selection = self.device_tree.selection()
        if selection:
            item_id = selection[0]
            device_id = self.device_tree.item(item_id, "text")
            
            # Toggle checkbox for multi-device mode
            if self.multi_device_mode:
                values = list(self.device_tree.item(item_id, "values"))
                if len(values) > 0:
                    # Toggle checkbox state
                    if values[0] == "☐":
                        values[0] = "☑️"
                    else:
                        values[0] = "☐"
                    self.device_tree.item(item_id, values=values)
                    
                    self._update_button_states()
            else:
                # Legacy single-device mode
                if hasattr(self, 'devices_list'):
                    for device in self.devices_list:
                        if device['id'] in device_id:  # Handle star marker
                            self.selected_device = device
                            self.connect_btn.configure(state="normal")
                            break
    
    def _update_button_states(self):
        """Update button states based on device connections"""
        connected_devices = self.multi_device_manager.get_connected_devices()
        
        # Enable/disable buttons based on connection state
        if connected_devices:
            self.disconnect_all_btn.configure(state="normal")
        else:
            self.disconnect_all_btn.configure(state="disabled")
        
        # Refresh device control widgets when selection changes
        self._refresh_device_control_widgets()
    
    def _start_bot(self):
        """Start the bot"""
        if not self.game_detector:
            messagebox.showwarning("No Connection", "Please connect to a device first.")
            return
        
        def bot_thread():
            try:
                self.bot_running = True
                self._update_status("🤖 Bot running...")
                
                # Update GUI
                self.message_queue.put({
                    'type': 'bot_started'
                })
                
                # Main bot loop
                while self.bot_running:
                    # Check game status
                    is_running, package = self.game_detector.is_lineage2m_running()
                    
                    if is_running:
                        game_state = self.game_detector.detect_game_state()
                        
                        self.message_queue.put({
                            'type': 'game_status_update',
                            'running': True,
                            'package': package,
                            'state': game_state
                        })
                    else:
                        self.message_queue.put({
                            'type': 'game_status_update',
                            'running': False,
                            'package': None,
                            'state': None
                        })
                    
                    # Take screenshot periodically
                    if self.bot_running:
                        screenshot = self.device_manager.adb.take_screenshot()
                        if screenshot is not None:
                            # Remove old screenshots if queue is full
                            try:
                                self.screenshot_queue.put_nowait(screenshot)
                            except queue.Full:
                                # Remove oldest screenshot and add new one
                                try:
                                    old_screenshot = self.screenshot_queue.get_nowait()
                                    del old_screenshot  # Explicitly delete old screenshot
                                    self.screenshot_queue.put_nowait(screenshot)
                                except queue.Empty:
                                    pass
                    
                    # Wait for configured interval
                    time.sleep(self.interval_var.get())
                
            except Exception as e:
                logger.error(f"Error in bot thread: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Bot error: {e}"
                })
            finally:
                self.bot_running = False
                self.message_queue.put({
                    'type': 'bot_stopped'
                })
        
        # Start bot thread
        threading.Thread(target=bot_thread, daemon=True).start()
    
    def _stop_bot(self):
        """Stop the bot"""
        self.bot_running = False
        self._update_status("⏹️ Stopping bot...")
    
    def _take_screenshot(self):
        """Take a manual screenshot"""
        if not self.device_manager or not self.device_manager.adb.connected:
            messagebox.showwarning("No Connection", "Please connect to a device first.")
            return
        
        def screenshot_thread():
            try:
                self._update_status("📸 Taking screenshot...")
                
                screenshot = self.device_manager.adb.take_screenshot()
                if screenshot is not None:
                    # Release old screenshot if exists
                    if hasattr(self, 'current_screenshot') and self.current_screenshot is not None:
                        del self.current_screenshot
                    
                    # Add to queue (remove oldest if full)
                    try:
                        self.screenshot_queue.put_nowait(screenshot)
                    except queue.Full:
                        try:
                            old_screenshot = self.screenshot_queue.get_nowait()
                            del old_screenshot
                            self.screenshot_queue.put_nowait(screenshot)
                        except queue.Empty:
                            pass
                    
                    self.current_screenshot = screenshot
                    
                    self.message_queue.put({
                        'type': 'screenshot_taken'
                    })
                else:
                    raise BotError("Failed to capture screenshot")
                    
            except Exception as e:
                logger.error(f"Error taking screenshot: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Screenshot failed: {e}"
                })
        
        threading.Thread(target=screenshot_thread, daemon=True).start()
    
    # Per-device control methods
    def _start_device_bot(self, device_id):
        """Start bot for a specific device - automatically launches game if not running"""
        if device_id not in self.multi_device_manager.get_connected_devices():
            messagebox.showwarning("Device Not Connected", f"Device {device_id} is not connected.")
            return
        
        # Check game state and launch automatically if needed
        def check_and_start():
            try:
                self._update_status(f"🔍 Checking game state for {device_id}...")
                
                # Get device session
                with self.multi_device_manager.lock:
                    if device_id not in self.multi_device_manager.connected_devices:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Device {device_id} session not found"))
                        return
                    
                    session = self.multi_device_manager.connected_devices[device_id]
                
                # Check if game is running
                game_detector = None
                if hasattr(self, 'game_detector') and self.game_detector:
                    # Use existing game detector if available
                    game_detector = self.game_detector
                else:
                    # Create game detector for this device
                    from ..modules.game_detector import GameDetector
                    game_detector = GameDetector(session.adb, self.config.game)
                
                is_running, package = game_detector.is_lineage2m_running()
                
                # If game is not running, launch it automatically
                if not is_running:
                    self._update_status(f"🚀 Game not running, launching automatically for {device_id}...")
                    logger.info(f"Game not running for {device_id}, launching automatically")
                    
                    # Get installed packages
                    installed_packages = game_detector.get_installed_lineage2m_packages()
                    if not installed_packages:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"No Lineage 2M packages found on device {device_id}"))
                        self.root.after(0, lambda: self._update_status(f"❌ No Lineage 2M packages found on {device_id}"))
                        return
                    
                    # Use the first installed package (should be the variant like lineage2mnu)
                    target_package = installed_packages[0]
                    logger.info(f"Launching game package: {target_package} for {device_id}")
                    
                    # Launch the game
                    launch_success = game_detector.launch_game(target_package)
                    
                    if not launch_success:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to launch game on {device_id}. Check logs for details."))
                        self.root.after(0, lambda: self._update_status(f"❌ Failed to launch game on {device_id}"))
                        return
                    
                    # Wait a bit for game to start (optional, but helps ensure game is ready)
                    self._update_status(f"⏳ Waiting for game to start on {device_id}...")
                    time.sleep(3)  # Wait 3 seconds for game to start
                    
                    # Verify game started (optional check)
                    is_running_new, package_new = game_detector.is_lineage2m_running()
                    if is_running_new:
                        logger.info(f"Game successfully launched and verified running: {package_new}")
                        self._update_status(f"✅ Game launched successfully on {device_id}")
                    else:
                        logger.warning(f"Game launch command sent but game not detected yet on {device_id}")
                        self._update_status(f"⚠️ Game launch command sent on {device_id}, proceeding anyway...")
                
                # Game is running (or was just launched), start the bot
                self.root.after(0, lambda: self._actually_start_device_bot(device_id))
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error checking game state for {device_id}: {error_msg}", exc_info=True)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Error starting bot: {msg}"))
                self.root.after(0, lambda msg=error_msg: self._update_status(f"❌ Error starting bot: {msg}"))
        
        # Run check in background thread
        threading.Thread(target=check_and_start, daemon=True).start()
    
    def _show_game_state_dialog_and_start(self, device_id, is_running, package, game_state, game_detector, screenshot=None, foreground_app=None, installed_packages=None):
        """Show game state dialog where user can manually decide if game is running"""
        try:
            # Create custom dialog window
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Game State - Decide Before Starting Bot")
            dialog.geometry("600x500")
            dialog.transient(self.root)
            dialog.grab_set()  # Make dialog modal
            
            # Store user's decision
            user_decision = {'value': None}
            
            # Main frame
            main_frame = ctk.CTkFrame(dialog)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Title
            title_label = ctk.CTkLabel(
                main_frame,
                text="🎮 Game State Decision",
                font=ctk.CTkFont(size=18, weight="bold")
            )
            title_label.pack(pady=(10, 5))
            
            device_label = ctk.CTkLabel(
                main_frame,
                text=f"Device: {device_id}",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            device_label.pack(pady=(0, 20))
            
            # Detected status frame
            detected_frame = ctk.CTkFrame(main_frame)
            detected_frame.pack(fill="x", padx=10, pady=5)
            
            detected_label = ctk.CTkLabel(
                detected_frame,
                text="🔍 System Detection:",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            detected_label.pack(pady=5)
            
            if is_running:
                detected_status = f"✅ Detected: Game Running\n📦 Package: {package}"
                if game_state:
                    detected_status += f"\n📊 Menu State: {game_state.get('menu_state', 'unknown')}\n"
                    detected_status += f"   UI Elements: {game_state.get('ui_elements', 0)}"
            else:
                detected_status = f"❌ Detected: Game Not Running"
                if foreground_app:
                    detected_status += f"\n📱 Current App: {foreground_app}"
            
            detected_text = ctk.CTkLabel(
                detected_frame,
                text=detected_status,
                font=ctk.CTkFont(size=12),
                justify="left"
            )
            detected_text.pack(pady=5, padx=10)
            
            # User decision frame
            decision_frame = ctk.CTkFrame(main_frame)
            decision_frame.pack(fill="x", padx=10, pady=10)
            
            decision_label = ctk.CTkLabel(
                decision_frame,
                text="👤 Your Decision:",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            decision_label.pack(pady=10)
            
            # Radio buttons for decision
            decision_var = tk.StringVar(value="detected")  # Default to detected value
            
            radio_running = ctk.CTkRadioButton(
                decision_frame,
                text="✅ Game IS Running",
                variable=decision_var,
                value="running",
                font=ctk.CTkFont(size=12)
            )
            radio_running.pack(pady=5, padx=20, anchor="w")
            
            radio_not_running = ctk.CTkRadioButton(
                decision_frame,
                text="❌ Game is NOT Running",
                variable=decision_var,
                value="not_running",
                font=ctk.CTkFont(size=12)
            )
            radio_not_running.pack(pady=5, padx=20, anchor="w")
            
            # Set default based on detection
            if is_running:
                decision_var.set("running")
            else:
                decision_var.set("not_running")
            
            # Launch game button (only show if game is not running)
            launch_button_frame = ctk.CTkFrame(main_frame)
            launch_button_frame.pack(fill="x", padx=10, pady=5)
            
            launch_status_label = ctk.CTkLabel(
                launch_button_frame,
                text="",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            launch_status_label.pack(pady=5)
            
            def on_launch_game():
                """Launch the game automatically"""
                launch_status_label.configure(text="🚀 Launching game...", text_color="blue")
                launch_btn.configure(state="disabled")
                dialog.update()
                
                def launch_in_thread():
                    try:
                        logger.info(f"Launch button clicked for device: {device_id}")
                        logger.info(f"Game detector available: {game_detector is not None}")
                        logger.info(f"Package passed: {package}")
                        logger.info(f"Installed packages: {installed_packages}")
                        
                        # Launch game using game detector
                        if game_detector:
                            # Check ADB connection status
                            if hasattr(game_detector.adb, 'device_id'):
                                logger.info(f"ADB device_id: {game_detector.adb.device_id}")
                                logger.info(f"ADB connected: {game_detector.adb.connected}")
                            
                            # Use package if available, otherwise let game_detector find one
                            target_package = package
                            
                            # If no package specified, prefer variant packages (like lineage2mnu) over base packages
                            if not target_package and installed_packages:
                                # Prefer packages that are NOT in the default list (variants like lineage2mnu)
                                variant_packages = [p for p in installed_packages if 'lineage2mnu' in p or 'lineage2m.global' in p or 'lineage2m.sea' in p or 'lineage2m.kr' in p]
                                if variant_packages:
                                    target_package = variant_packages[0]
                                    logger.info(f"Using variant package: {target_package}")
                                else:
                                    target_package = installed_packages[0]
                            
                            if not target_package:
                                logger.error("No package specified and no installed packages found")
                                dialog.after(0, lambda: launch_status_label.configure(
                                    text="❌ No package found. Cannot launch game.", 
                                    text_color="red"
                                ))
                                dialog.after(0, lambda: launch_btn.configure(state="normal"))
                                return
                            
                            logger.info(f"Attempting to launch package: {target_package}")
                            logger.info(f"ACTUAL ADB COMMAND will be: adb -s {game_detector.adb.device_id} shell monkey -p {target_package} 1")
                            success = game_detector.launch_game(target_package)
                            logger.info(f"Launch result: {success}")
                            
                            # Schedule UI update on main thread
                            dialog.after(0, lambda s=success: on_launch_complete(s))
                        else:
                            logger.error("Game detector is None")
                            dialog.after(0, lambda: launch_status_label.configure(text="❌ Game detector not available", text_color="red"))
                            dialog.after(0, lambda: launch_btn.configure(state="normal"))
                    except Exception as e:
                        logger.error(f"Error launching game: {e}", exc_info=True)
                        dialog.after(0, lambda: launch_status_label.configure(text=f"❌ Error: {e}", text_color="red"))
                        dialog.after(0, lambda: launch_btn.configure(state="normal"))
                
                def on_launch_complete(success):
                    if success:
                        launch_status_label.configure(text="✅ Launch command executed! Checking if game started...", text_color="green")
                        logger.info(f"Game launch command sent for {device_id}")
                        
                        # Wait a bit and check if game started
                        dialog.after(3000, check_game_after_launch)
                    else:
                        launch_status_label.configure(
                            text="❌ All launch methods failed. Check logs for details.\nTry manually launching the game.", 
                            text_color="red"
                        )
                        launch_btn.configure(state="normal")
                        logger.error(f"Game launch failed for {device_id}. Check logs above for command details.")
                
                # Run launch in background thread
                threading.Thread(target=launch_in_thread, daemon=True).start()
            
            def check_game_after_launch():
                """Check if game started after launch command"""
                try:
                    is_running_new, package_new = game_detector.is_lineage2m_running()
                    if is_running_new:
                        launch_status_label.configure(text="✅ Game is now running!", text_color="green")
                        decision_var.set("running")
                        detected_text.configure(
                            text=f"✅ Detected: Game Running\n📦 Package: {package_new}\n\n✅ Game launched successfully!"
                        )
                        launch_btn.configure(state="normal")
                        logger.info(f"Game successfully launched and verified running: {package_new}")
                    else:
                        # Check how many times we've tried (max 5 attempts = 15 seconds)
                        if not hasattr(check_game_after_launch, 'attempts'):
                            check_game_after_launch.attempts = 0
                        check_game_after_launch.attempts += 1
                        
                        if check_game_after_launch.attempts < 5:
                            launch_status_label.configure(
                                text=f"⏳ Checking game status... ({check_game_after_launch.attempts}/5)", 
                                text_color="orange"
                            )
                            # Try again after another delay
                            dialog.after(3000, check_game_after_launch)
                        else:
                            launch_status_label.configure(
                                text="⚠️ Game may not have started. Check device screen.\nYou can still start bot manually.", 
                                text_color="orange"
                            )
                            launch_btn.configure(state="normal")
                            logger.warning(f"Game launch command sent but game not detected after 15 seconds")
                except Exception as e:
                    logger.error(f"Error checking game after launch: {e}")
                    launch_status_label.configure(text="⚠️ Could not verify game status", text_color="orange")
                    launch_btn.configure(state="normal")
            
            if not is_running:
                # Show available packages if multiple
                if installed_packages and len(installed_packages) > 1:
                    packages_text = f"Available packages: {', '.join(installed_packages)}"
                    packages_label = ctk.CTkLabel(
                        launch_button_frame,
                        text=packages_text,
                        font=ctk.CTkFont(size=10),
                        text_color="gray"
                    )
                    packages_label.pack(pady=2)
                elif installed_packages:
                    packages_text = f"Package: {installed_packages[0]}"
                    packages_label = ctk.CTkLabel(
                        launch_button_frame,
                        text=packages_text,
                        font=ctk.CTkFont(size=10),
                        text_color="gray"
                    )
                    packages_label.pack(pady=2)
                
                launch_btn = ctk.CTkButton(
                    launch_button_frame,
                    text="🚀 Launch Game Automatically",
                    command=on_launch_game,
                    width=200,
                    fg_color="blue"
                )
                launch_btn.pack(pady=5)
            
            # Button frame
            button_frame = ctk.CTkFrame(main_frame)
            button_frame.pack(fill="x", padx=10, pady=20)
            
            def on_start():
                user_decision['value'] = decision_var.get()
                dialog.destroy()
            
            def on_cancel():
                user_decision['value'] = None
                dialog.destroy()
            
            start_btn = ctk.CTkButton(
                button_frame,
                text="▶️ Start Bot",
                command=on_start,
                width=150,
                fg_color="green"
            )
            start_btn.pack(side="left", padx=10, pady=10)
            
            cancel_btn = ctk.CTkButton(
                button_frame,
                text="❌ Cancel",
                command=on_cancel,
                width=150
            )
            cancel_btn.pack(side="right", padx=10, pady=10)
            
            # Wait for dialog to close
            dialog.wait_window()
            
            # Process user decision
            decision = user_decision['value']
            
            if decision is None:
                # User cancelled
                self._update_status(f"❌ Bot start cancelled by user for {device_id}")
                return
            
            # Log the decision
            if decision == "running":
                logger.info(f"User decided: Game IS running for {device_id}")
                self._update_status(f"✅ Starting bot - User confirmed game is running ({device_id})")
            else:
                logger.info(f"User decided: Game is NOT running for {device_id}")
                self._update_status(f"⚠️ Starting bot - User confirmed game is NOT running ({device_id})")
            
            # Start the bot with user's decision
            self._actually_start_device_bot(device_id)
                
        except Exception as e:
            logger.error(f"Error showing game state dialog: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error showing game state: {e}")
    
    def _actually_start_device_bot(self, device_id):
        """Actually start bot for a specific device (called after state confirmation)"""
        def device_bot_thread():
            try:
                # Mark device bot as running
                if device_id in self.device_control_widgets:
                    self.device_control_widgets[device_id]['running'] = True
                
                self._update_status(f"🤖 Starting bot for {device_id}...")
                
                # Update button states
                self.message_queue.put({
                    'type': 'device_bot_started',
                    'device_id': device_id
                })
                
                # Register device with state monitor
                device_state_monitor.register_device(device_id)
                device_state_monitor.update_bot_state(device_id, True, "Starting")
                
                # Initialize game automation for this device
                try:
                    # Get device session
                    connected_devices = self.multi_device_manager.get_connected_devices()
                    if device_id not in connected_devices:
                        logger.error(f"Device {device_id} not in connected devices")
                        device_state_monitor.update_bot_state(device_id, False, error="Device not connected")
                        return
                    
                    device_session = self.multi_device_manager.connected_devices[device_id]
                    adb_manager = device_session.adb
                    
                    # Create game detector for this device
                    game_detector = GameDetector(adb_manager, self.config.game)
                    
                    # Create game automation with device ID for state monitoring
                    game_automation = GameAutomation(adb_manager, game_detector, device_id=device_id)
                    game_automation.start()
                    
                    # Store game automation for accessing player parameters
                    self.device_game_automations[device_id] = game_automation
                    
                    logger.info(f"Game automation initialized for device {device_id}")
                    
                except Exception as e:
                    logger.error(f"Error initializing game automation for {device_id}: {e}", exc_info=True)
                    device_state_monitor.update_bot_state(device_id, False, error=str(e))
                    return
                
                # Device-specific bot loop
                try:
                    while (device_id in self.device_control_widgets and 
                           self.device_control_widgets[device_id]['running']):
                        
                        # Execute bot commands on specific device
                        try:
                            # Check if device is still connected
                            if device_id not in self.multi_device_manager.get_connected_devices():
                                logger.warning(f"Device {device_id} disconnected, stopping bot")
                                device_state_monitor.update_bot_state(device_id, False, error="Device disconnected")
                                break
                            
                            # Check game state
                            is_running, package = game_detector.is_lineage2m_running()
                            logger.debug(f"Device {device_id} - Game running check: {is_running}, package: {package}")
                            
                            if is_running:
                                game_state = game_detector.detect_game_state()
                                game_state_str = game_state.get('status', 'unknown')
                                tap_screen_detected = game_state.get('tap_screen_detected', False)
                                
                                logger.debug(f"Device {device_id} - Game state detected: {game_state_str}, tap_screen: {tap_screen_detected}")
                                
                                # If "Tap screen" is detected, set state to "select_server" and ensure bot is running
                                if tap_screen_detected:
                                    game_state_str = 'select_server'
                                    # Update bot state to running if it's not already
                                    current_bot_state = device_state_monitor.get_device_state(device_id)
                                    if current_bot_state and not current_bot_state.bot_state.is_running:
                                        device_state_monitor.update_bot_state(device_id, True, "Waiting for server selection")
                                
                                # Get current game state to preserve actual_game_state and detailed_game_state
                                current_state = device_state_monitor.get_device_state(device_id)
                                current_actual = current_state.game_state.actual_game_state if current_state else None
                                current_detailed = current_state.game_state.detailed_game_state if current_state else None
                                
                                device_state_monitor.update_game_state(
                                    device_id,
                                    is_running=True,
                                    package_name=package,
                                    game_state=game_state_str,
                                    actual_game_state=current_actual,
                                    detailed_game_state=current_detailed
                                )
                            else:
                                logger.debug(f"Device {device_id} - Game not detected as running")
                                device_state_monitor.update_game_state(device_id, is_running=False)
                                
                                # Also check using check_game_status for more reliable detection
                                try:
                                    game_packages = self.config.game.packages
                                    game_status = adb_manager.check_game_status(device_id, game_packages)
                                    if game_status.get('game_running'):
                                        running_packages = game_status.get('running_packages', [])
                                        if running_packages:
                                            logger.info(f"Device {device_id} - Game detected via check_game_status: {running_packages[0]}")
                                            
                                            # Check for "Tap screen" even if package check didn't detect it
                                            try:
                                                screenshot = adb_manager.take_screenshot()
                                                if screenshot is not None:
                                                    tap_screen_detected = game_detector._detect_tap_screen_text(screenshot)
                                                    game_state_str = 'select_server' if tap_screen_detected else 'detected'
                                                    
                                                    if tap_screen_detected:
                                                        # Update bot state to running
                                                        device_state_monitor.update_bot_state(device_id, True, "Waiting for server selection")
                                                else:
                                                    game_state_str = 'detected'
                                            except Exception as e:
                                                logger.debug(f"Error checking tap screen text: {e}")
                                                game_state_str = 'detected'
                                            
                                            # Get current game state to preserve actual_game_state and detailed_game_state
                                            current_state = device_state_monitor.get_device_state(device_id)
                                            current_actual = current_state.game_state.actual_game_state if current_state else None
                                            current_detailed = current_state.game_state.detailed_game_state if current_state else None
                                            
                                            device_state_monitor.update_game_state(
                                                device_id,
                                                is_running=True,
                                                package_name=running_packages[0],
                                                game_state=game_state_str,
                                                actual_game_state=current_actual,
                                                detailed_game_state=current_detailed
                                            )
                                except Exception as e:
                                    logger.debug(f"Error checking game status via check_game_status: {e}")
                            
                            # Run game automation loop
                            game_automation.run_game_loop()
                            
                            # Update bot state with last action
                            game_state = game_automation.get_game_state()
                            current_game_state = game_state.get('actual_game_state', 'unknown')
                            detailed_game_state = game_state.get('detailed_game_state', 'unknown')
                            
                            # Update device state monitor with actual_game_state and detailed_game_state
                            device_state_monitor.update_game_state(
                                device_id,
                                is_running=True,
                                actual_game_state=current_game_state,
                                detailed_game_state=detailed_game_state
                            )
                            
                            if current_game_state == 'playing':
                                device_state_monitor.update_bot_state(device_id, True, "Playing")
                            elif current_game_state in ['select_server', 'select_character']:
                                device_state_monitor.update_bot_state(device_id, True, "In Menu")
                            else:
                                device_state_monitor.update_bot_state(device_id, True, "Monitoring")
                            
                            # Take screenshot from this device
                            result = self.multi_device_manager.execute_on_device(
                                device_id, 
                                'take_screenshot'
                            )
                            
                            # Check if result is not None (result is a numpy array, so use 'is not None' instead of truthy check)
                            if result is not None:
                                self.screenshot_queue.put(result)
                            
                        except Exception as e:
                            logger.error(f"Error in device bot for {device_id}: {e}", exc_info=True)
                            device_state_monitor.update_bot_state(device_id, True, error=str(e))
                            time.sleep(1)  # Wait a bit before retrying
                            continue
                        
                        # Wait for configured interval, but check running flag periodically
                        # This allows the bot to stop more responsively
                        interval = self.interval_var.get()
                        elapsed = 0
                        check_interval = 0.5  # Check every 0.5 seconds
                        while elapsed < interval:
                            if (device_id not in self.device_control_widgets or 
                                not self.device_control_widgets[device_id]['running']):
                                break
                            sleep_time = min(check_interval, interval - elapsed)
                            time.sleep(sleep_time)
                            elapsed += sleep_time
                
                finally:
                    # Stop game automation when bot stops
                    try:
                        game_automation.stop()
                        logger.info(f"Game automation stopped for device {device_id}")
                        
                        # Remove game automation from storage
                        if device_id in self.device_game_automations:
                            del self.device_game_automations[device_id]
                    except Exception as e:
                        logger.error(f"Error stopping game automation: {e}")
                
            except Exception as e:
                logger.error(f"Error in device bot thread for {device_id}: {e}")
            finally:
                # Mark device bot as stopped
                if device_id in self.device_control_widgets:
                    self.device_control_widgets[device_id]['running'] = False
                
                # Update state monitor
                device_state_monitor.update_bot_state(device_id, False, "Stopped")
                
                self.message_queue.put({
                    'type': 'device_bot_stopped',
                    'device_id': device_id
                })
        
        threading.Thread(target=device_bot_thread, daemon=True).start()
    
    def _stop_device_bot(self, device_id):
        """Stop bot for a specific device"""
        try:
            logger.info(f"Stopping bot for device: {device_id}")
            self._update_status(f"⏹️ Stopping bot for {device_id}...")
            
            # Set running flag to False to stop the bot loop
            if device_id in self.device_control_widgets:
                self.device_control_widgets[device_id]['running'] = False
                logger.info(f"Set running flag to False for device: {device_id}")
            
            # Send message to update button states immediately
            self.message_queue.put({
                'type': 'device_bot_stopped',
                'device_id': device_id
            })
            
            logger.info(f"Bot stop command sent for device: {device_id}")
            
        except Exception as e:
            logger.error(f"Error stopping bot for {device_id}: {e}", exc_info=True)
            self._update_status(f"❌ Error stopping bot for {device_id}: {e}")
    
    def _check_game_status(self, device_id):
        """Check and display game status for a specific device"""
        if device_id not in self.multi_device_manager.get_connected_devices():
            messagebox.showwarning("Device Not Connected", f"Device {device_id} is not connected.")
            return
        
        def check_thread():
            try:
                self._update_status(f"🔍 Checking game status for {device_id}...")
                
                # Get device session
                device_session = self.multi_device_manager.connected_devices[device_id]
                adb_manager = device_session.adb
                
                # Create game detector
                game_detector = GameDetector(adb_manager, self.config.game)
                
                # Check using is_lineage2m_running
                is_running, package = game_detector.is_lineage2m_running()
                
                # Also check using check_game_status
                game_packages = self.config.game.packages
                game_status = adb_manager.check_game_status(device_id, game_packages)
                
                # Get foreground app
                foreground_app = adb_manager.get_foreground_app()
                
                # Check for "Tap screen" text
                tap_screen_detected = False
                try:
                    screenshot = adb_manager.take_screenshot()
                    if screenshot is not None:
                        tap_screen_detected = game_detector._detect_tap_screen_text(screenshot)
                except Exception as e:
                    logger.debug(f"Error checking tap screen text: {e}")
                
                # Build status message
                status_lines = [
                    f"Game Status Check for {device_id}",
                    "=" * 50,
                    f"Foreground App: {foreground_app or 'Unknown'}",
                    "",
                    f"is_lineage2m_running(): {is_running}",
                    f"Package: {package or 'None'}",
                    "",
                    f"check_game_status():",
                    f"  Game Running: {game_status.get('game_running', False)}",
                    f"  Game Installed: {game_status.get('game_installed', False)}",
                    f"  Installed Packages: {game_status.get('installed_packages', [])}",
                    f"  Running Packages: {game_status.get('running_packages', [])}",
                    f"  Foreground Package: {game_status.get('foreground_package', 'None')}",
                    "",
                    f"'Tap screen' detected: {tap_screen_detected}",
                    "",
                    "=" * 50
                ]
                
                status_text = "\n".join(status_lines)
                
                # Update state monitor
                if is_running or game_status.get('game_running'):
                    game_state_str = 'select_server' if tap_screen_detected else 'detected'
                    
                    # If "Tap screen" is detected, set bot state to running
                    if tap_screen_detected:
                        device_state_monitor.update_bot_state(device_id, True, "Waiting for server selection")
                    
                    device_state_monitor.update_game_state(
                        device_id,
                        is_running=True,
                        package_name=package or game_status.get('running_packages', [None])[0],
                        game_state=game_state_str
                    )
                else:
                    device_state_monitor.update_game_state(device_id, is_running=False)
                
                # Show result
                self.root.after(0, lambda: messagebox.showinfo("Game Status Check", status_text))
                self._update_status(f"✅ Game status checked for {device_id}")
                
            except Exception as e:
                error_msg = f"Error checking game status: {e}"
                logger.error(error_msg, exc_info=True)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        threading.Thread(target=check_thread, daemon=True).start()
    
    def _take_device_screenshot(self, device_id):
        """Take screenshot from a specific device"""
        if device_id not in self.multi_device_manager.get_connected_devices():
            messagebox.showwarning("Device Not Connected", f"Device {device_id} is not connected.")
            return
        
        def screenshot_thread():
            try:
                self._update_status(f"📸 Taking screenshot from {device_id}...")
                
                result = self.multi_device_manager.execute_on_device(
                    device_id, 
                    'take_screenshot'
                )
                
                # Check if result is not None (result is a numpy array, so use 'is not None' instead of truthy check)
                if result is not None:
                    self.screenshot_queue.put(result)
                    self.message_queue.put({
                        'type': 'screenshot_taken',
                        'device_id': device_id
                    })
                else:
                    raise BotError(f"Failed to capture screenshot from {device_id}")
                    
            except Exception as e:
                logger.error(f"Error taking screenshot from {device_id}: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Screenshot failed for {device_id}: {e}"
                })
        
        threading.Thread(target=screenshot_thread, daemon=True).start()
    
    def _device_tap(self, device_id):
        """Perform tap gesture at center of device screen"""
        if device_id not in self.multi_device_manager.get_connected_devices():
            self.root.after(0, lambda: messagebox.showwarning("Device Not Connected", f"Device {device_id} is not connected."))
            return
        
        def tap_thread():
            try:
                # Get device session
                with self.multi_device_manager.lock:
                    if device_id not in self.multi_device_manager.connected_devices:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Device {device_id} session not found"))
                        return
                    
                    session = self.multi_device_manager.connected_devices[device_id]
                
                # Get device resolution
                logger.info(f"Getting screen resolution for device {device_id}")
                resolution_result = session.adb.execute_adb_command(['shell', 'wm', 'size'])
                
                if not resolution_result[0]:
                    error_msg = f"Failed to get screen resolution for {device_id}"
                    logger.error(error_msg)
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    self.root.after(0, lambda: self._update_status(f"❌ {error_msg}"))
                    return
                
                # Parse resolution from output like "Physical size: 1080x1920"
                resolution_text = resolution_result[1].strip()
                logger.info(f"Device {device_id} resolution output: {resolution_text}")
                
                # Extract width and height from resolution string
                width, height = None, None
                if 'Physical size:' in resolution_text:
                    # Format: "Physical size: 1080x1920"
                    size_part = resolution_text.split('Physical size:')[1].strip()
                    if 'x' in size_part:
                        width, height = map(int, size_part.split('x'))
                elif 'x' in resolution_text:
                    # Direct format: "1080x1920"
                    size_part = resolution_text.split()[0] if ' ' in resolution_text else resolution_text
                    width, height = map(int, size_part.split('x'))
                
                if width is None or height is None:
                    error_msg = f"Could not parse resolution for {device_id}: {resolution_text}"
                    logger.error(error_msg)
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    self.root.after(0, lambda: self._update_status(f"❌ {error_msg}"))
                    return
                
                # Calculate center coordinates
                center_x = width // 2
                center_y = height // 2
                
                logger.info(f"Device {device_id} - Resolution: {width}x{height}, Center: ({center_x}, {center_y})")
                
                # Update status
                self.root.after(0, lambda: self._update_status(f"👆 Tapping center ({center_x},{center_y}) on {device_id}..."))
                
                # Execute tap at center using direct session command
                tap_result = session.execute_command(['shell', 'input', 'tap', str(center_x), str(center_y)])
                
                if tap_result[0]:
                    success_msg = f"✅ Tapped center ({center_x},{center_y}) on {device_id}"
                    logger.info(f"Tap successful: {success_msg}")
                    self.root.after(0, lambda: self._update_status(success_msg))
                else:
                    error_msg = f"❌ Tap failed on {device_id}: {tap_result[1]}"
                    logger.error(error_msg)
                    self.root.after(0, lambda: self._update_status(error_msg))
                    self.root.after(0, lambda: messagebox.showerror("Tap Failed", f"Tap command failed on {device_id}:\n{tap_result[1]}"))
                    
            except Exception as e:
                error_msg = f"Error performing tap on {device_id}: {e}"
                logger.error(error_msg, exc_info=True)
                self.root.after(0, lambda: self._update_status(f"❌ {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        threading.Thread(target=tap_thread, daemon=True).start()
    
    def _device_swipe(self, device_id):
        """Perform swipe gesture on specific device"""
        if device_id not in self.multi_device_manager.get_connected_devices():
            messagebox.showwarning("Device Not Connected", f"Device {device_id} is not connected.")
            return
        
        # Simple dialog to get swipe coordinates
        dialog = ctk.CTkInputDialog(text=f"Enter swipe coordinates for {device_id} (x1,y1,x2,y2):", title="Swipe Coordinates")
        coordinates = dialog.get_input()
        
        if coordinates:
            try:
                x1, y1, x2, y2 = map(int, coordinates.split(','))
                
                def swipe_thread():
                    try:
                        result = self.multi_device_manager.execute_on_device(
                            device_id,
                            'swipe',
                            x1=x1, y1=y1, x2=x2, y2=y2
                        )
                        
                        if result:
                            self._update_status(f"👈 Swiped from ({x1},{y1}) to ({x2},{y2}) on {device_id}")
                        else:
                            self._update_status(f"❌ Swipe failed on {device_id}")
                            
                    except Exception as e:
                        logger.error(f"Error performing swipe on {device_id}: {e}")
                        self._update_status(f"❌ Swipe error on {device_id}: {e}")
                
                threading.Thread(target=swipe_thread, daemon=True).start()
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter coordinates in format: x1,y1,x2,y2")
    
    def _device_test_actions(self, device_id):
        """Perform automated test sequence of swipe and tap actions on specific device"""
        # Immediate feedback - show that button was clicked with device address
        device_address = str(device_id)
        print(f"TEST BUTTON CLICKED for device address: {device_address}")
        logger.info(f"Test button clicked for device address: {device_address}")
        
        # Store device_id in a local variable to avoid closure issues
        dev_id = str(device_id)
        
        # Update status immediately - use a simple function call
        def update_status_safe(msg):
            try:
                self._update_status(msg)
            except Exception as e:
                logger.error(f"Error updating status: {e}")
        
        try:
            self.root.after(0, lambda: update_status_safe(f"🧪 Test button clicked for device: {dev_id}"))
        except Exception as e:
            logger.error(f"Error scheduling status update: {e}")
        
        # Start thread to check connection and run test - don't block GUI
        def check_and_run():
            try:
                # Check if device is connected (this might take time, so do in thread)
                connected_devices = self.multi_device_manager.get_connected_devices()
                logger.info(f"Connected devices: {list(connected_devices.keys())}")
                logger.info(f"Testing device address: {dev_id}")
                logger.info(f"Checking if {dev_id} is in connected devices...")
                
                if dev_id not in connected_devices:
                    logger.warning(f"Device address {dev_id} not found in connected devices")
                    error_msg = f"Device address {dev_id} is not connected.\n\nConnected devices: {list(connected_devices.keys())}"
                    self.root.after(0, lambda: messagebox.showwarning("Device Not Connected", error_msg))
                    self.root.after(0, lambda: update_status_safe(f"❌ Device {dev_id} not connected"))
                    return
                
                logger.info(f"Device address {dev_id} is connected and ready for testing")
                
                # Get device session to verify it's using the correct address
                with self.multi_device_manager.lock:
                    if dev_id in self.multi_device_manager.connected_devices:
                        session = self.multi_device_manager.connected_devices[dev_id]
                        actual_device_id = session.adb.device_id if session.adb.device_id else "NOT SET"
                        logger.info(f"DeviceSession device_id: {session.device_id}, ADBManager device_id: {actual_device_id}")
                        
                        if actual_device_id != dev_id:
                            logger.warning(f"Device ID mismatch! Expected {dev_id}, ADBManager has {actual_device_id}")
                            # Try to fix it
                            if session.adb.connect_to_device(dev_id):
                                logger.info(f"Fixed ADBManager device_id to {dev_id}")
                            else:
                                logger.error(f"Failed to fix ADBManager device_id")
                
                # Update status with device address confirmation
                self.root.after(0, lambda: update_status_safe(f"🧪 Testing device: {dev_id} - Starting test sequence..."))
                
                # Define test sequence function
                def test_sequence():
                    def safe_update_status(text):
                        """Thread-safe status update"""
                        try:
                            self.root.after(0, lambda t=text: update_status_safe(t))
                        except Exception as e:
                            logger.error(f"Error scheduling status update: {e}")
                    
                    def take_and_display_screenshot(action_name=""):
                        """Take screenshot and display it in GUI"""
                        try:
                            screenshot = self.multi_device_manager.execute_on_device(
                                dev_id, 'take_screenshot'
                            )
                            if screenshot is not None:
                                # Put screenshot in queue for GUI display
                                self.screenshot_queue.put(screenshot)
                                logger.info(f"Screenshot captured for {action_name} on {dev_id}")
                                return True
                            else:
                                logger.warning(f"Failed to capture screenshot for {action_name}")
                                return False
                        except Exception as e:
                            logger.error(f"Error taking screenshot: {e}")
                            return False
                    
                    try:
                        safe_update_status(f"🧪 Starting test sequence on device: {dev_id}")
                        logger.info(f"Test sequence thread started for device address: {dev_id}")
                        
                        # Verify device is still connected before starting
                        connected_devices = self.multi_device_manager.get_connected_devices()
                        if dev_id not in connected_devices:
                            error_msg = f"Device {dev_id} disconnected before test could start"
                            logger.error(error_msg)
                            safe_update_status(f"❌ {error_msg}")
                            return
                        
                        # Get device session to verify device address
                        session = None
                        with self.multi_device_manager.lock:
                            if dev_id in self.multi_device_manager.connected_devices:
                                session = self.multi_device_manager.connected_devices[dev_id]
                                logger.info(f"Using device session for address: {dev_id}")
                                logger.info(f"Session device_id: {session.device_id}, ADBManager device_id: {session.adb.device_id}")
                            else:
                                logger.error(f"Device session not found for {dev_id}")
                                safe_update_status(f"❌ Device session not found for {dev_id}")
                                return
                        
                        # Helper function to run ADB commands directly (like test_adb_simple.py)
                        def run_adb_direct(device_id, cmd_parts):
                            """Run ADB command directly using subprocess (same as test_adb_simple.py)"""
                            full_cmd = ['adb', '-s', device_id] + cmd_parts
                            full_cmd_str = ' '.join(full_cmd)
                            logger.info(f"ACTUAL ADB COMMAND: {full_cmd_str}")
                            
                            try:
                                result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
                                logger.info(f"ADB command result - returncode={result.returncode}")
                                if result.stdout:
                                    logger.info(f"ADB command stdout: {result.stdout.strip()}")
                                if result.stderr:
                                    logger.warning(f"ADB command stderr: {result.stderr.strip()}")
                                
                                success = result.returncode == 0
                                output = result.stdout if success else result.stderr
                                return success, output
                            except subprocess.TimeoutExpired:
                                logger.error(f"ADB command timeout: {full_cmd_str}")
                                return False, "Command timeout"
                            except Exception as e:
                                logger.error(f"ADB command exception: {full_cmd_str}, error: {e}")
                                return False, str(e)
                        
                        # Get device resolution first to verify coordinates (using direct subprocess)
                        try:
                            safe_update_status(f"📱 Device {dev_id}: Getting resolution...")
                            resolution_success, resolution_output = run_adb_direct(dev_id, ['shell', 'wm', 'size'])
                            if resolution_success:
                                resolution_text = resolution_output.strip()
                                safe_update_status(f"📱 Device {dev_id} - Resolution: {resolution_text}")
                                logger.info(f"Device address {dev_id} resolution: {resolution_text}")
                            else:
                                logger.warning(f"Could not get device resolution for {dev_id}: {resolution_output}")
                                safe_update_status(f"⚠️ Device {dev_id}: Could not get resolution")
                        except Exception as e:
                            logger.warning(f"Exception getting device resolution for {dev_id}: {e}")
                        
                        # Verify ADB connection with test command (using direct subprocess)
                        try:
                            safe_update_status(f"🔍 Device {dev_id}: Testing connection...")
                            test_success, test_output = run_adb_direct(dev_id, ['shell', 'echo', f'Testing device {dev_id}'])
                            if not test_success:
                                logger.error(f"Connection test failed for {dev_id}: {test_output}")
                                safe_update_status(f"❌ Connection test failed for {dev_id}")
                                return
                            else:
                                logger.info(f"Connection test succeeded for {dev_id}")
                                safe_update_status(f"✅ Device {dev_id}: Connection OK")
                        except Exception as e:
                            logger.error(f"Connection test exception for {dev_id}: {e}")
                            safe_update_status(f"❌ Connection test error for {dev_id}: {e}")
                            return
                        
                        # Take initial screenshot
                        safe_update_status(f"📸 Device {dev_id}: Taking initial screenshot...")
                        take_and_display_screenshot("initial")
                        time.sleep(0.5)
                        
                        # 1. Tap center of screen (using direct subprocess like test_adb_simple.py)
                        logger.info(f"Testing tap at center (540, 960) on device address: {dev_id}")
                        safe_update_status(f"👆 Device {dev_id}: Tapping center (540,960)...")
                        time.sleep(1)  # Wait like test_adb_simple.py
                        try:
                            center_tap_success, center_tap_output = run_adb_direct(dev_id, ['shell', 'input', 'tap', '540', '960'])
                            
                            if center_tap_success:
                                logger.info(f"[OK] Tap command executed successfully on {dev_id}!")
                                safe_update_status(f"✅ Device {dev_id}: Tap executed at center (540,960) - Check device screen!")
                            else:
                                logger.error(f"[FAIL] Tap command failed on {dev_id}: {center_tap_output}")
                                safe_update_status(f"❌ Device {dev_id}: Tap failed - {center_tap_output}")
                            
                            # Wait a bit for device to respond, then take screenshot
                            time.sleep(0.8)
                            if take_and_display_screenshot("after center tap"):
                                if center_tap_success:
                                    safe_update_status(f"✅ Device {dev_id}: Tap executed - Check screenshot!")
                                else:
                                    safe_update_status(f"⚠️ Device {dev_id}: Tap may have failed - Check screenshot!")
                        except Exception as e:
                            logger.error(f"Exception during tap on device {dev_id}: {e}", exc_info=True)
                            safe_update_status(f"❌ Device {dev_id}: Test tap exception - {e}")
                        
                        time.sleep(0.5)
                        
                        # 2. Test tap at alternative center (960, 540) - for different resolution (like test_adb_simple.py)
                        logger.info(f"Testing tap at alternative center (960, 540) - adjusted for 1920x1080 on device {dev_id}")
                        safe_update_status(f"👆 Device {dev_id}: Tapping alternative center (960,540)...")
                        time.sleep(2)  # Wait like test_adb_simple.py
                        try:
                            alt_tap_success, alt_tap_output = run_adb_direct(dev_id, ['shell', 'input', 'tap', '960', '540'])
                            
                            if alt_tap_success:
                                logger.info(f"[OK] Alternative tap executed successfully on {dev_id}!")
                                safe_update_status(f"✅ Device {dev_id}: Alternative tap executed (960,540) - Check device screen!")
                            else:
                                logger.error(f"[FAIL] Alternative tap failed on {dev_id}: {alt_tap_output}")
                            
                            time.sleep(0.8)
                            if take_and_display_screenshot("after alternative tap"):
                                if alt_tap_success:
                                    safe_update_status(f"✅ Device {dev_id}: Alternative tap executed - Check screenshot!")
                        except Exception as e:
                            logger.error(f"Exception during alternative tap on device {dev_id}: {e}", exc_info=True)
                            safe_update_status(f"❌ Device {dev_id}: Alternative tap exception - {e}")
                        
                        time.sleep(0.5)
                        
                        # 3. Test swipe (using direct subprocess like test_adb_simple.py)
                        logger.info(f"Testing swipe on device {dev_id}")
                        safe_update_status(f"👈 Device {dev_id}: Swiping...")
                        time.sleep(1)  # Wait like test_adb_simple.py
                        try:
                            swipe_success, swipe_output = run_adb_direct(dev_id, ['shell', 'input', 'swipe', '300', '540', '700', '540', '300'])
                            
                            if swipe_success:
                                logger.info(f"[OK] Swipe executed successfully on {dev_id}!")
                                safe_update_status(f"✅ Device {dev_id}: Swipe executed - Check device screen!")
                            else:
                                logger.error(f"[FAIL] Swipe failed on {dev_id}: {swipe_output}")
                                safe_update_status(f"❌ Device {dev_id}: Swipe failed - {swipe_output}")
                            
                            time.sleep(0.8)
                            if take_and_display_screenshot("after swipe"):
                                if swipe_success:
                                    safe_update_status(f"✅ Device {dev_id}: Swipe executed - Check screenshot!")
                        except Exception as e:
                            logger.error(f"Exception during swipe on device {dev_id}: {e}", exc_info=True)
                            safe_update_status(f"❌ Device {dev_id}: Swipe exception - {e}")
                        
                        
                        # Final screenshot
                        time.sleep(0.5)
                        safe_update_status(f"📸 Device {dev_id}: Taking final screenshot...")
                        take_and_display_screenshot("final")
                        
                        # Final status
                        safe_update_status(f"🎯 Device {dev_id}: Test sequence completed! Check Monitor tab for screenshots")
                        logger.info(f"Test sequence completed successfully on device address: {dev_id}")
                        logger.info("="*60)
                        logger.info("Test completed!")
                        logger.info("="*60)
                        logger.info("NOTE: If taps don't appear on screen:")
                        logger.info("1. Make sure screen is unlocked")
                        logger.info("2. Check if app is in foreground")
                        logger.info("3. Verify coordinates match device resolution")
                        logger.info("4. Try different coordinates")
                        
                    except Exception as e:
                        logger.error(f"Error during test sequence on {dev_id}: {e}", exc_info=True)
                        safe_update_status(f"❌ Test sequence error on {dev_id}: {e}")
                        error_msg = f"Test sequence failed on {dev_id}:\n{e}"
                        self.root.after(0, lambda: messagebox.showerror("Test Error", error_msg))
                
                # Run test sequence in background thread
                logger.info(f"Creating test sequence thread for {dev_id}...")
                thread = threading.Thread(target=test_sequence, daemon=True, name=f"TestSequence-{dev_id}")
                thread.start()
                logger.info(f"Test sequence thread started for {dev_id}")
                
            except Exception as e:
                logger.error(f"Error in check_and_run for {dev_id}: {e}", exc_info=True)
                self.root.after(0, lambda: update_status_safe(f"❌ Error: {e}"))
                self.root.after(0, lambda: messagebox.showerror("Test Error", f"Error starting test: {e}"))
        
        # Start the check_and_run in a thread to avoid blocking GUI
        try:
            check_thread = threading.Thread(target=check_and_run, daemon=True, name=f"CheckAndRun-{dev_id}")
            check_thread.start()
        except Exception as e:
            logger.error(f"Error starting check thread: {e}", exc_info=True)
            self.root.after(0, lambda: update_status_safe(f"❌ Failed to start test: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Test Error", f"Failed to start test thread: {e}"))
    
    # Global control methods
    def _start_all_bots(self):
        """Start bots for all selected devices"""
        selected_devices = self._get_selected_devices_from_tree()
        connected_devices = self.multi_device_manager.get_connected_devices()
        
        for device in selected_devices:
            device_id = device['id']
            if device_id in connected_devices:
                self._start_device_bot(device_id)
    
    def _stop_all_bots(self):
        """Stop bots for all selected devices"""
        for device_id in list(self.device_control_widgets.keys()):
            self._stop_device_bot(device_id)
    
    def _take_screenshot_all(self):
        """Take screenshots from all selected devices"""
        selected_devices = self._get_selected_devices_from_tree()
        connected_devices = self.multi_device_manager.get_connected_devices()
        
        for device in selected_devices:
            device_id = device['id']
            if device_id in connected_devices:
                self._take_device_screenshot(device_id)
    
    def _update_interval_label(self, value):
        """Update the interval label when slider changes"""
        self.interval_label.configure(text=f"{float(value):.1f}s")
    
    def _clear_logs(self):
        """Clear the log display"""
        self.log_text.delete("1.0", tk.END)
    
    def _save_logs(self):
        """Save logs to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                log_content = self.log_text.get("1.0", tk.END)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                
                self._update_status(f"💾 Logs saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save logs: {e}")
    
    def _load_config_display(self):
        """Load configuration into the display"""
        try:
            config_dict = {
                'adb': {
                    'timeout': self.config.adb.timeout,
                    'screenshot_timeout': self.config.adb.screenshot_timeout,
                    'connection_retry_count': self.config.adb.connection_retry_count
                },
                'game': {
                    'packages': self.config.game.packages,
                    'detection_interval': self.config.game.detection_interval,
                    'auto_launch': self.config.game.auto_launch
                },
                'touch': {
                    'tap_duration': self.config.touch.tap_duration,
                    'swipe_duration': self.config.touch.swipe_duration,
                    'coordinate_scaling': self.config.touch.coordinate_scaling
                },
                'logging': {
                    'level': self.config.logging.level,
                    'file_enabled': self.config.logging.file_enabled,
                    'console_enabled': self.config.logging.console_enabled
                }
            }
            
            config_yaml = yaml.dump(config_dict, default_flow_style=False, indent=2)
            self.config_text.delete("1.0", tk.END)
            self.config_text.insert("1.0", config_yaml)
            
        except Exception as e:
            logger.error(f"Error loading config display: {e}")
    
    def _reload_config(self):
        """Reload configuration from file"""
        try:
            # Reload config manager
            config_manager._load_config()
            self.config = config_manager.get_config()
            
            # Update display
            self._load_config_display()
            
            self._update_status("🔄 Configuration reloaded")
            
        except Exception as e:
            messagebox.showerror("Config Error", f"Failed to reload config: {e}")
    
    def _save_config(self):
        """Save configuration from the display"""
        try:
            # Get config text
            config_yaml = self.config_text.get("1.0", tk.END)
            config_dict = yaml.safe_load(config_yaml)
            
            # Update config manager (this is simplified - full implementation would update all fields)
            if 'game' in config_dict and 'detection_interval' in config_dict['game']:
                self.interval_var.set(config_dict['game']['detection_interval'])
            
            # Save to file
            config_manager.save_config()
            
            self._update_status("💾 Configuration saved")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save config: {e}")
    
    def _open_config_file(self):
        """Open the configuration file in default editor"""
        try:
            config_path = config_manager.config_file
            if config_path.exists():
                # Open with default application
                if hasattr(subprocess, 'run'):
                    subprocess.run(['notepad.exe', str(config_path)], check=False)
                else:
                    webbrowser.open(str(config_path))
            else:
                messagebox.showwarning("File Not Found", f"Config file not found: {config_path}")
                
        except Exception as e:
            messagebox.showerror("Open Error", f"Failed to open config file: {e}")
    
    def _save_screenshot(self):
        """Save the current screenshot"""
        if self.current_screenshot is None:
            messagebox.showwarning("No Screenshot", "No screenshot available to save.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            
            if file_path:
                cv2.imwrite(file_path, self.current_screenshot)
                self._update_status(f"📸 Screenshot saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save screenshot: {e}")
    
    def _test_connection(self):
        """Test the current connection"""
        if not self.device_manager or not self.device_manager.adb.connected:
            messagebox.showwarning("No Connection", "Please connect to a device first.")
            return
        
        def test_thread():
            try:
                self._update_status("🧪 Testing connection...")
                
                # Test screenshot
                screenshot = self.device_manager.adb.take_screenshot()
                if screenshot is not None:
                    # Test app detection
                    foreground_app = self.device_manager.adb.get_foreground_app()
                    
                    # Test game detection
                    if self.game_detector:
                        is_running, package = self.game_detector.is_lineage2m_running()
                        
                        test_results = {
                            'screenshot': True,
                            'app_detection': foreground_app is not None,
                            'game_detection': is_running,
                            'foreground_app': foreground_app,
                            'game_package': package if is_running else None
                        }
                    else:
                        test_results = {
                            'screenshot': True,
                            'app_detection': foreground_app is not None,
                            'game_detection': None,
                            'foreground_app': foreground_app,
                            'game_package': None
                        }
                    
                    self.message_queue.put({
                        'type': 'test_results',
                        'results': test_results
                    })
                else:
                    raise BotError("Screenshot test failed")
                    
            except Exception as e:
                logger.error(f"Connection test error: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Connection test failed: {e}"
                })
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """Lineage 2M Bot - Advanced Automation Framework
        
Version: 1.0.0
Author: Lineage2M Bot Team

A comprehensive automation framework for Lineage 2M mobile game using ADB, computer vision, and advanced bot capabilities.

Features:
• Device discovery and management
• Game state detection and monitoring
• Screenshot capture and analysis
• Modern GUI interface
• Configurable automation settings
• Advanced logging system

For support and documentation, visit the project repository."""
        
        messagebox.showinfo("About Lineage 2M Bot", about_text)
    
    def _process_message(self, message):
        """Process messages from background threads"""
        msg_type = message.get('type')
        
        if msg_type == 'devices_discovered':
            device_status = message.get('device_status', {})
            self._update_device_list(message['devices'])
            
            # Include device limit info in status
            connected_count = device_status.get('connected_count', 0)
            max_devices = device_status.get('max_devices', 0)
            status_text = f"✅ Found {len(message['devices'])} device(s)"
            if max_devices > 0:
                status_text += f" (Connected: {connected_count}/{max_devices})"
            
            self._update_status(status_text)
            self._update_button_states()
            
        elif msg_type == 'devices_discovered_and_connected':
            devices = message['devices']
            connected_count = message['connected_count']
            device_status = message.get('device_status', {})
            max_devices = device_status.get('max_devices', 0)
            
            self._update_device_list(devices)
            
            # Auto-select (check) connected devices in the device tree
            connected_devices = self.multi_device_manager.get_connected_devices()
            self._auto_select_connected_devices(connected_devices)
            
            # Enhanced status with device limits
            status_text = f"✅ Discovered {len(devices)} device(s), auto-connected to {connected_count} game-ready devices"
            if max_devices > 0:
                status_text += f" (Limit: {max_devices})"
            
            self._update_status(status_text)
            
            # Update connection status with limit info
            connection_text = f"🎮 Auto-connected: {connected_count}"
            if max_devices > 0:
                connection_text += f"/{max_devices}"
            connection_text += " game devices"
            
            self.connection_status.configure(text=connection_text)
            
            # Refresh connection display and update buttons
            self._refresh_device_connections()
            self._update_button_states()
            
            # Refresh per-device control widgets after auto-selection
            self._refresh_device_control_widgets()
            
        elif msg_type == 'device_added_manually':
            device_id = message['device_id']
            devices = message['devices']
            device_status = message.get('device_status', {})
            max_devices = device_status.get('max_devices', 0)
            connected_count = device_status.get('connected_count', 0)
            
            self._update_device_list(devices)
            
            # Update region tab device list
            if hasattr(self, '_update_region_device_list'):
                self.root.after(0, self._update_region_device_list)
            
            # Auto-select the newly added device
            connected_devices = self.multi_device_manager.get_connected_devices()
            self._auto_select_connected_devices(connected_devices)
            
            # Update status
            status_text = f"✅ Successfully added device: {device_id}"
            if max_devices > 0:
                status_text += f" (Connected: {connected_count}/{max_devices})"
            
            self._update_status(status_text)
            
            # Update connection status
            if max_devices > 0:
                connection_text = f"🟢 Connected: {connected_count}/{max_devices}"
            else:
                connection_text = f"🟢 Connected: {connected_count} device(s)"
            self.connection_status.configure(text=connection_text)
            
            # Refresh connection display and update buttons
            self._refresh_device_connections()
            self._update_button_states()
            
            # Refresh per-device control widgets
            self._refresh_device_control_widgets()
            
        elif msg_type == 'devices_restored':
            # Update region tab device list
            if hasattr(self, '_update_region_device_list'):
                self.root.after(0, self._update_region_device_list)
            devices = message['devices']
            restored_count = message.get('restored_count', 0)
            device_status = message.get('device_status', {})
            max_devices = device_status.get('max_devices', 0)
            connected_count = device_status.get('connected_count', 0)
            
            self._update_device_list(devices)
            
            # Auto-select restored devices
            connected_devices = self.multi_device_manager.get_connected_devices()
            self._auto_select_connected_devices(connected_devices)
            
            # Update status
            status_text = f"✅ Restored {restored_count} saved device(s)"
            if max_devices > 0:
                status_text += f" (Connected: {connected_count}/{max_devices})"
            
            self._update_status(status_text)
            
            # Update connection status
            if max_devices > 0:
                connection_text = f"🟢 Connected: {connected_count}/{max_devices}"
            else:
                connection_text = f"🟢 Connected: {connected_count} device(s)"
            self.connection_status.configure(text=connection_text)
            
            # Refresh connection display and update buttons
            self._refresh_device_connections()
            self._update_button_states()
            
            # Refresh per-device control widgets
            self._refresh_device_control_widgets()
            
        elif msg_type == 'update_device_list':
            # Update device list (including failed saved devices)
            devices = message.get('devices', [])
            if devices:
                self._update_device_list(devices)
                self._update_button_states()
        
        elif msg_type == 'devices_restore_failed':
            message_text = message.get('message', 'Could not restore saved devices')
            self._update_status(f"⚠️ {message_text}")
            
        elif msg_type == 'device_connected':
            device = message['device']
            self.connection_status.configure(text=f"🟢 Connected: {device['id']}")
            self._update_status(f"✅ Connected to {device['id']}")
            
            # Enable bot controls
            self.start_bot_btn.configure(state="normal")
            self.screenshot_btn.configure(state="normal")
            
        elif msg_type == 'multi_devices_connected':
            connected_count = message['connected_count']
            failed_devices = message.get('failed_devices', [])
            
            if failed_devices:
                status_text = f"✅ Connected to {connected_count} devices, {len(failed_devices)} failed"
            else:
                status_text = f"✅ Connected to {connected_count} devices"
            
            self._update_status(status_text)
            self.connection_status.configure(text=f"🟢 Multi-device: {connected_count} connected")
            
            # Update device tree to show connection status
            self._refresh_device_connections()
            self._update_button_states()
            
            # Refresh device control widgets
            self._refresh_device_control_widgets()
            
        elif msg_type == 'all_devices_disconnected':
            self._update_status("🚫 Disconnected from all devices")
            self.connection_status.configure(text="🔴 Not connected")
            self._refresh_device_connections()
            self._update_button_states()
            
            # Refresh device control widgets
            self._refresh_device_control_widgets()
            
        elif msg_type == 'bot_started':
            self.start_bot_btn.configure(state="disabled")
            self.stop_bot_btn.configure(state="normal")
            
        elif msg_type == 'bot_stopped':
            self.start_bot_btn.configure(state="normal")
            self.stop_bot_btn.configure(state="disabled")
            self._update_status("⏹️ Bot stopped")
            
        elif msg_type == 'game_status_update':
            self._update_game_status(message)
            
        elif msg_type == 'screenshot_taken':
            device_id = message.get('device_id')
            if device_id:
                self._update_status(f"📸 Screenshot captured from {device_id}")
            else:
                self._update_status("📸 Screenshot captured")
            
        elif msg_type == 'device_bot_started':
            device_id = message['device_id']
            if device_id in self.device_control_widgets:
                widgets = self.device_control_widgets[device_id]
                widgets['start_btn'].configure(state="disabled")
                widgets['stop_btn'].configure(state="normal")
            self._update_status(f"🤖 Bot started for {device_id}")
            
        elif msg_type == 'device_bot_stopped':
            device_id = message['device_id']
            if device_id in self.device_control_widgets:
                widgets = self.device_control_widgets[device_id]
                widgets['start_btn'].configure(state="normal")
                widgets['stop_btn'].configure(state="disabled")
            self._update_status(f"⏹️ Bot stopped for {device_id}")
            
        elif msg_type == 'test_results':
            self._show_test_results(message['results'])
            
        elif msg_type == 'error':
            messagebox.showerror("Error", message['message'])
            self._update_status(f"❌ Error: {message['message']}")
    
    def _update_device_list(self, devices):
        """Update the device list display"""
        # Clear existing items
        for item in self.device_tree.get_children():
            self.device_tree.delete(item)
        
        # Deduplicate devices by device_id before processing
        seen_device_ids = set()
        unique_devices = []
        for device in devices:
            device_id = device['id']
            if device_id not in seen_device_ids:
                seen_device_ids.add(device_id)
                unique_devices.append(device)
            else:
                logger.warning(f"Duplicate device found in devices list: {device_id}, skipping")
        
        # Separate and prioritize devices with games
        game_devices = []
        regular_devices = []
        
        for device in unique_devices:
            game_status = device.get('game_status', {})
            if game_status.get('installed') or game_status.get('running'):
                game_devices.append(device)
            else:
                regular_devices.append(device)
        
        # Add game devices first (prioritized)
        for device in game_devices:
            self._add_device_to_tree(device, is_game_device=True)
        
        # Then add regular devices
        for device in regular_devices:
            self._add_device_to_tree(device, is_game_device=False)
    
    def _add_device_to_tree(self, device, is_game_device=False):
        """Add a device to the tree view with multi-device support"""
        game_status = device.get('game_status', {})
        device_id = device['id']
        
        # Check if this device already exists in the tree (prevent duplicates)
        for item in self.device_tree.get_children():
            existing_device_id = self.device_tree.item(item, "text").replace(" ⭐", "").replace(" 💾", "")
            if existing_device_id == device_id:
                logger.warning(f"Device {device_id} already exists in tree, skipping duplicate")
                return
        
        # Check if device is connected in multi-device manager
        connected_devices = self.multi_device_manager.get_connected_devices()
        is_connected_multi = device_id in connected_devices
        
        # Check for connection error message (for saved devices that failed to connect)
        connection_error = device.get('connection_error')
        device_status_value = device.get('status', 'unknown')
        
        if connection_error:
            connection_status = f"🔴 {connection_error}"
        elif device_status_value == 'pending':
            connection_status = "🟡 Connecting..."
        elif is_connected_multi:
            connection_status = "🟢 Connected"
        else:
            connection_status = "🔴 Disconnected"
        
        # Determine device status and icon
        device_status_value = device.get('status', 'unknown')
        
        if is_game_device:
            # Game devices are automatically marked as "available" for bot use
            if game_status.get('running'):
                status_icon = "🎮"
                device_status = "available"
            elif game_status.get('installed'):
                status_icon = "📱"  
                device_status = "available"
            else:
                if device_status_value == 'pending':
                    status_icon = "🟡"
                    device_status = "connecting"
                elif device_status_value == 'connected':
                    status_icon = "🟢"
                    device_status = "available"
                else:
                    status_icon = "🔴"
                    device_status = "unavailable"
        else:
            # Regular devices - show available/unavailable based on connection status
            if device_status_value == 'pending':
                status_icon = "🟡"
                device_status = "connecting"
            elif device_status_value == 'connected':
                status_icon = "🟢"
                device_status = "available"
            else:
                status_icon = "🔴"
                device_status = "unavailable"
        
        # Game status information
        game_info = "Not installed"
        if game_status.get('running'):
            running_count = len(game_status.get('running_packages', []))
            game_info = f"🎮 Running ({running_count})"
        elif game_status.get('installed'):
            installed_count = len(game_status.get('installed_packages', []))
            game_info = f"📱 Installed ({installed_count})"
        
        # Add priority indicator for game devices
        # Add saved device indicator if it's a saved device that failed to connect
        display_device_id = device_id
        if is_game_device:
            display_device_id += " ⭐"
        if device.get('is_saved_device') and not is_connected_multi:
            display_device_id += " 💾"  # Indicate this is a saved device
        
        # Default checkbox state
        checkbox_state = "☐"
        
        self.device_tree.insert(
            "",
            "end",
            text=display_device_id,
            values=(
                checkbox_state,  # Checkbox
                device.get('type', 'Unknown'),
                device.get('model', 'Unknown'),
                device.get('android_version', 'Unknown'),
                device.get('resolution', 'Unknown'),
                f"{status_icon} {device_status}",
                game_info,
                connection_status
            )
        )
    
    def _update_game_status(self, message):
        """Update the game status display"""
        if message['running']:
            status_text = f"🎮 Game Running: {message['package']}\n"
            if message['state']:
                state = message['state']
                status_text += f"Screen Size: {state.get('screen_size', 'Unknown')}\n"
                status_text += f"Menu State: {state.get('menu_state', 'Unknown')}\n"
                status_text += f"UI Elements: {state.get('ui_elements', 0)}\n"
                status_text += f"Last Update: {state.get('timestamp', 'Unknown')}\n"
        else:
            status_text = "❌ Lineage 2M not detected\n"
            
            # Show current foreground app
            foreground_app = self.device_manager.adb.get_foreground_app()
            if foreground_app:
                status_text += f"Current App: {foreground_app}\n"
        
        # Update text area
        self.game_status_text.delete("1.0", tk.END)
        self.game_status_text.insert("1.0", status_text)
        
        # Add to log
        self._add_log(status_text.strip())
    
    def _update_screenshot_display(self, screenshot):
        """Update the screenshot display (removed - no longer used)"""
        # This function is kept for compatibility but does nothing
        # The live screenshot feature has been removed from the monitor tab
        pass
    
    def _show_test_results(self, results):
        """Show connection test results"""
        result_text = "🧪 Connection Test Results:\n\n"
        result_text += f"Screenshot: {'✅' if results['screenshot'] else '❌'}\n"
        result_text += f"App Detection: {'✅' if results['app_detection'] else '❌'}\n"
        
        if results['game_detection'] is not None:
            result_text += f"Game Detection: {'✅' if results['game_detection'] else '❌'}\n"
        
        if results['foreground_app']:
            result_text += f"Current App: {results['foreground_app']}\n"
        
        if results['game_package']:
            result_text += f"Game Package: {results['game_package']}\n"
        
        messagebox.showinfo("Test Results", result_text)
        self._update_status("🧪 Connection test completed")
    
    def _add_log(self, text):
        """Add text to the log display"""
        try:
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.log_text.see(tk.END)  # Scroll to bottom
        except Exception as e:
            logger.error(f"Error adding log: {e}")
    
    def _update_status(self, text):
        """Update the status bar"""
        try:
            self.status_label.configure(text=text)
            self._add_log(text)
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def _show_progress(self):
        """Show progress bar"""
        try:
            self.progress_bar.pack(side="right", padx=10, pady=5)
            self.progress_bar.start()
        except Exception as e:
            logger.error(f"Error showing progress: {e}")
    
    def _hide_progress(self):
        """Hide progress bar"""
        try:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
        except Exception as e:
            logger.error(f"Error hiding progress: {e}")