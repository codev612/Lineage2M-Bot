"""
Device State Monitor - Tracks bot control state and game playing state per device
"""

import time
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BotControlState:
    """Bot control state for a device"""
    is_running: bool = False
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_action: Optional[str] = None
    last_action_time: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class GamePlayingState:
    """Game playing state for a device"""
    is_running: bool = False
    package_name: Optional[str] = None
    game_state: str = "unknown"  # in_game, main_menu, loading_screen, etc.
    actual_game_state: Optional[str] = None  # Actual game state: 'select_server', 'select_character', 'playing', 'unknown', etc.
    detailed_game_state: Optional[str] = None  # Detailed game state: 'auto_questing', 'purchasing', 'quest_ended', etc.
    last_detected: Optional[datetime] = None
    state_changes: int = 0
    last_action: Optional[str] = None  # auto_attack, collect, potion, etc.
    action_count: Dict[str, int] = field(default_factory=dict)
    health_status: Optional[str] = None  # healthy, low_hp, low_mp
    detected_items: int = 0
    quest_status: Optional[str] = None


@dataclass
class DeviceState:
    """Complete state for a device"""
    device_id: str
    bot_state: BotControlState = field(default_factory=BotControlState)
    game_state: GamePlayingState = field(default_factory=GamePlayingState)
    last_updated: datetime = field(default_factory=datetime.now)
    is_connected: bool = False


class DeviceStateMonitor:
    """
    Monitors bot control state and game playing state for each device
    """
    
    def __init__(self):
        """Initialize device state monitor"""
        self.device_states: Dict[str, DeviceState] = {}
        self.lock = threading.Lock()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.update_callback = None
        
        logger.info("Device state monitor initialized")
    
    def register_device(self, device_id: str):
        """Register a device for monitoring"""
        with self.lock:
            if device_id not in self.device_states:
                self.device_states[device_id] = DeviceState(device_id=device_id)
                logger.info(f"Registered device for monitoring: {device_id}")
    
    def unregister_device(self, device_id: str):
        """Unregister a device from monitoring"""
        with self.lock:
            if device_id in self.device_states:
                del self.device_states[device_id]
                logger.info(f"Unregistered device from monitoring: {device_id}")
    
    def update_bot_state(self, device_id: str, is_running: bool, action: str = None, error: str = None):
        """Update bot control state for a device"""
        with self.lock:
            if device_id not in self.device_states:
                self.register_device(device_id)
            
            state = self.device_states[device_id]
            bot_state = state.bot_state
            
            if is_running != bot_state.is_running:
                if is_running:
                    bot_state.is_running = True
                    bot_state.started_at = datetime.now()
                    bot_state.stopped_at = None
                    logger.info(f"Bot started for device {device_id}")
                else:
                    bot_state.is_running = False
                    bot_state.stopped_at = datetime.now()
                    logger.info(f"Bot stopped for device {device_id}")
            
            if action:
                bot_state.last_action = action
                bot_state.last_action_time = datetime.now()
            
            if error:
                bot_state.error_count += 1
                bot_state.last_error = error
                logger.warning(f"Bot error for device {device_id}: {error}")
            
            state.last_updated = datetime.now()
    
    def update_game_state(self, device_id: str, 
                         is_running: bool = None,
                         package_name: str = None,
                         game_state: str = None,
                         actual_game_state: str = None,
                         detailed_game_state: str = None,
                         action: str = None,
                         health_status: str = None,
                         detected_items: int = None,
                         quest_status: str = None):
        """Update game playing state for a device"""
        with self.lock:
            if device_id not in self.device_states:
                self.register_device(device_id)
            
            state = self.device_states[device_id]
            game_state_obj = state.game_state
            
            if is_running is not None:
                if is_running != game_state_obj.is_running:
                    game_state_obj.state_changes += 1
                game_state_obj.is_running = is_running
                game_state_obj.last_detected = datetime.now()
            
            if package_name:
                game_state_obj.package_name = package_name
            
            if game_state:
                if game_state != game_state_obj.game_state:
                    game_state_obj.state_changes += 1
                game_state_obj.game_state = game_state
            
            # Always update actual_game_state and detailed_game_state if provided (including 'unknown')
            if actual_game_state is not None:
                game_state_obj.actual_game_state = actual_game_state
            # If not provided but we want to keep existing value, don't change it
            
            if detailed_game_state is not None:
                game_state_obj.detailed_game_state = detailed_game_state
            # If not provided but we want to keep existing value, don't change it
            
            if action:
                game_state_obj.last_action = action
                if action not in game_state_obj.action_count:
                    game_state_obj.action_count[action] = 0
                game_state_obj.action_count[action] += 1
            
            if health_status:
                game_state_obj.health_status = health_status
            
            if detected_items is not None:
                game_state_obj.detected_items = detected_items
            
            if quest_status:
                game_state_obj.quest_status = quest_status
            
            state.last_updated = datetime.now()
    
    def update_connection_state(self, device_id: str, is_connected: bool):
        """Update connection state for a device"""
        with self.lock:
            if device_id not in self.device_states:
                self.register_device(device_id)
            
            self.device_states[device_id].is_connected = is_connected
            self.device_states[device_id].last_updated = datetime.now()
    
    def get_device_state(self, device_id: str) -> Optional[DeviceState]:
        """Get current state for a device"""
        with self.lock:
            return self.device_states.get(device_id)
    
    def get_all_states(self) -> Dict[str, DeviceState]:
        """Get all device states"""
        with self.lock:
            return self.device_states.copy()
    
    def set_update_callback(self, callback):
        """Set callback function to be called when states are updated"""
        self.update_callback = callback
    
    def start_monitoring(self, interval: float = 2.0):
        """Start monitoring thread"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # Check for stale states (devices not updated in 30 seconds)
                    current_time = datetime.now()
                    stale_devices = []
                    
                    with self.lock:
                        for device_id, state in self.device_states.items():
                            time_diff = (current_time - state.last_updated).total_seconds()
                            if time_diff > 30 and state.is_connected:
                                stale_devices.append(device_id)
                    
                    # Call update callback if set
                    if self.update_callback:
                        try:
                            self.update_callback()
                        except Exception as e:
                            logger.error(f"Error in update callback: {e}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Device state monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Device state monitoring stopped")
    
    def get_state_summary(self, device_id: str) -> Dict[str, Any]:
        """
        Get a summary of device state for display
        
        Returns:
            Dictionary with formatted state information
        """
        state = self.get_device_state(device_id)
        if not state:
            return {
                'device_id': device_id,
                'bot_status': 'Unknown',
                'game_status': 'Unknown',
                'connection_status': 'Unknown'
            }
        
        # Bot status
        if state.bot_state.is_running:
            bot_status = f"🟢 Running"
            if state.bot_state.started_at:
                runtime = (datetime.now() - state.bot_state.started_at).total_seconds()
                bot_status += f" ({int(runtime//60)}m {int(runtime%60)}s)"
            if state.bot_state.last_action:
                bot_status += f" | Last: {state.bot_state.last_action}"
            if state.bot_state.error_count > 0:
                bot_status += f" | Errors: {state.bot_state.error_count}"
        else:
            bot_status = "🔴 Stopped"
            if state.bot_state.last_error:
                bot_status += f" | {state.bot_state.last_error[:30]}"
        
        # Game status
        if state.game_state.is_running:
            game_status = f"🟢 Running"
            if state.game_state.package_name:
                # Show short package name (last part after .)
                package_short = state.game_state.package_name.split('.')[-1]
                game_status += f" ({package_short})"
            # Show actual_game_state and detailed_game_state (always show if available)
            if state.game_state.actual_game_state:
                game_status += f" | Actual: {state.game_state.actual_game_state}"
            if state.game_state.detailed_game_state:
                game_status += f" | Detailed: {state.game_state.detailed_game_state}"
            if state.game_state.game_state and state.game_state.game_state != 'unknown':
                game_status += f" | State: {state.game_state.game_state}"
            if state.game_state.last_action:
                game_status += f" | Action: {state.game_state.last_action}"
            if state.game_state.health_status and state.game_state.health_status != 'healthy':
                game_status += f" | Health: {state.game_state.health_status}"
        else:
            game_status = "🔴 Not Running"
            if state.game_state.last_detected:
                # Show how long ago it was last detected
                time_since = (datetime.now() - state.game_state.last_detected).total_seconds()
                if time_since < 60:
                    game_status += f" (last seen {int(time_since)}s ago)"
                else:
                    game_status += f" (last seen {int(time_since//60)}m ago)"
        
        # Connection status
        connection_status = "🟢 Connected" if state.is_connected else "🔴 Disconnected"
        
        return {
            'device_id': device_id,
            'bot_status': bot_status,
            'game_status': game_status,
            'connection_status': connection_status,
            'bot_running': state.bot_state.is_running,
            'game_running': state.game_state.is_running,
            'game_state': state.game_state.game_state,
            'actual_game_state': state.game_state.actual_game_state,
            'detailed_game_state': state.game_state.detailed_game_state,
            'last_action': state.bot_state.last_action,
            'error_count': state.bot_state.error_count,
            'action_counts': state.game_state.action_count.copy()
        }

# Global device state monitor instance
device_state_monitor = DeviceStateMonitor()

