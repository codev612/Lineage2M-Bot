"""
Game Automation Module - Implements actual gameplay scenarios for Lineage 2M
Handles game actions, state detection, and automated gameplay strategies
"""

import time
import gc
import shutil
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import cv2
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import config_manager
from ..core.adb_manager import ADBManager
from .game_detector import GameDetector

# Import device state monitor (optional, to avoid circular import)
try:
    from ..utils.device_state_monitor import device_state_monitor
    STATE_MONITOR_AVAILABLE = True
except ImportError:
    STATE_MONITOR_AVAILABLE = False

logger = get_logger(__name__)

# Constants
DEFAULT_SCREEN_TRANSITION_TIME = 10.0  # Default time to wait for screen transitions (seconds)
MAX_RETRY_COUNT = 10  # Maximum number of retries for each step before stopping bot


class GameAutomation:
    """
    Handles automated gameplay scenarios for Lineage 2M
    Implements game actions, state detection, and gameplay strategies
    """
    
    def __init__(self, adb_manager: ADBManager, game_detector: GameDetector, device_id: str = None):
        """
        Initialize game automation
        
        Args:
            adb_manager: ADB manager instance for device control
            game_detector: Game detector instance for state detection
            device_id: Device ID for state monitoring (optional)
        """
        self.adb = adb_manager
        self.game_detector = game_detector
        self.config = config_manager.get_config()
        self.device_id = device_id
        self.running = False
        
        # Game state tracking (managed as dictionary/array)
        self.game_state = {
            'current_state': 'unknown',
            'running_state': 'unknown',  # 'not_running' or 'running'
            'actual_game_state': 'unknown',  # Actual game state from screenshot: 'select_server', 'select_character', 'playing', etc.
            'detailed_game_state': 'unknown',  # Detailed game state: 'auto_questing', etc. (more specific than actual_game_state)
            'last_updated': None
        }
        self.game_state_history = []  # List of state snapshots
        self.last_action_time = 0
        self.action_interval = 5.0  # Minimum time between actions (seconds)
        
        # Debug mode for detection (set to True to enable screenshot saving and detailed logging)
        self.debug_detection = True  # Set to False to disable debug features
        
        # Retry tracking for each step
        self.step_retry_count = {}  # Dictionary to track retry count per step
        self.max_retries = MAX_RETRY_COUNT
        
        # Screen dimensions (will be detected)
        self.screen_width = 0
        self.screen_height = 0
        
        # Screenshot caching to avoid multiple screenshots per loop
        self._last_screenshot = None
        self._last_screenshot_time = 0
        self._screenshot_cache_ttl = 0.5  # Cache screenshots for 0.5 seconds
        
        # Player parameters tracking
        self.player_parameters = {
            'blood_bottle': None,
            'bag_weight': None,
            'last_updated': None
        }
        self.player_parameters_history = []  # List of parameter snapshots
        
    def start(self):
        """Start the game automation"""
        if self.running:
            logger.warning("Game automation already running")
            return
        
        logger.info("Starting game automation...")
        self.running = True
        
        # Clear debug screenshots folder
        self._clear_debug_screenshots()
        
        # Get screen dimensions
        self._detect_screen_size()
        
    def stop(self):
        """Stop the game automation"""
        if not self.running:
            return
        
        logger.info("Stopping game automation...")
        self.running = False
        
        # Clear screenshot cache to free memory
        if self._last_screenshot is not None:
            del self._last_screenshot
            self._last_screenshot = None
            self._last_screenshot_time = 0
    
    def _clear_debug_screenshots(self):
        """Clear the debug_screenshots folder"""
        try:
            debug_dir = Path('debug_screenshots')
            if debug_dir.exists():
                # Count items before deletion
                items = list(debug_dir.iterdir())
                item_count = len(items)
                
                # Remove all files in the directory
                for file_path in items:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                
                if item_count > 0:
                    logger.info(f"Cleared debug_screenshots folder ({item_count} items removed)")
            else:
                # Create directory if it doesn't exist
                debug_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clear debug_screenshots folder: {e}")
    
    def _detect_screen_size(self):
        """Detect device screen dimensions"""
        screenshot = None
        try:
            screenshot = self.adb.take_screenshot()
            if screenshot is not None:
                self.screen_height, self.screen_width = screenshot.shape[:2]
                logger.info(f"Screen dimensions detected: {self.screen_width}x{self.screen_height}")
            else:
                # Fallback to ADB command
                success, output = self.adb.execute_adb_command(['shell', 'wm', 'size'])
                if success:
                    # Parse output like "Physical size: 1080x1920"
                    import re
                    match = re.search(r'(\d+)x(\d+)', output)
                    if match:
                        self.screen_width = int(match.group(1))
                        self.screen_height = int(match.group(2))
                        logger.info(f"Screen dimensions from ADB: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            logger.error(f"Error detecting screen size: {e}")
            # Default fallback
            self.screen_width = 1080
            self.screen_height = 1920
        finally:
            # Always release screenshot
            if screenshot is not None:
                del screenshot
                gc.collect(0)
    
    def _check_and_reset_retry(self, step_name: str) -> bool:
        """
        Check retry count for a step and increment it. Return True if should continue, False if should stop.
        
        Args:
            step_name: Name of the step being retried
            
        Returns:
            True if retries are within limit, False if max retries exceeded
        """
        if step_name not in self.step_retry_count:
            self.step_retry_count[step_name] = 0
        
        self.step_retry_count[step_name] += 1
        
        if self.step_retry_count[step_name] > self.max_retries:
            logger.error(f"Step '{step_name}' failed {self.max_retries} times. Stopping bot.")
            self.stop()
            return False
        
        logger.warning(f"Step '{step_name}' failed (attempt {self.step_retry_count[step_name]}/{self.max_retries})")
        return True
    
    def _reset_retry(self, step_name: str):
        """Reset retry count for a step after successful completion"""
        if step_name in self.step_retry_count:
            self.step_retry_count[step_name] = 0
    
    def _launch_game(self) -> bool:
        """
        Launch the game if it's not running
        
        Returns:
            True if launch was attempted/successful, False otherwise
        """
        try:
            # Get game packages from config
            game_packages = self.config.game.packages
            
            if not game_packages:
                logger.warning("No game packages configured")
                return False
            
            # Try to launch the first available package
            for package in game_packages:
                # Check if package is installed
                success, output = self.adb.execute_adb_command(['shell', 'pm', 'list', 'packages', package])
                if success and package in output:
                    # Package is installed, try to launch it
                    logger.info(f"Launching game package: {package}")
                    if self.adb.launch_app(package):
                        logger.info(f"Game launch command sent for {package}")
                        return True
                    else:
                        logger.warning(f"Failed to launch {package}")
            
            logger.warning("No installed game packages found to launch")
            return False
            
        except Exception as e:
            logger.error(f"Error launching game: {e}", exc_info=True)
            return False
    
    def run_game_loop(self):
        """
        Main game automation loop
        This is called repeatedly while the bot is running
        """
        if not self.running:
            return
        
        try:
            # Check if game is running
            is_running, package = self.game_detector.is_lineage2m_running()
            
            if not is_running:
                # Game is not running - set state to "not_running" and launch game
                self._update_game_state({
                    'running_state': 'not_running',
                    'current_state': 'not_running',
                    'actual_game_state': 'unknown'
                    # Note: detailed_game_state is independent and not reset here
                })
                
                logger.info("Game is not running, setting state to 'not_running' and attempting to launch...")
                step_name = "game_launch"
                
                if self._launch_game():
                    # Wait for game to start
                    logger.info(f"Game launch attempted, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s for game to start...")
                    time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                    
                    # Verify game is now running
                    is_running, package = self.game_detector.is_lineage2m_running()
                    if not is_running:
                        # Game still not running after launch attempt
                        if not self._check_and_reset_retry(step_name):
                            return  # Bot stopped due to max retries
                        logger.warning(f"Game launch attempted but still not running, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s...")
                        time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                        return
                    else:
                        # Success - reset retry counter
                        self._reset_retry(step_name)
                        logger.info("Game successfully launched")
                        # State will be updated to 'running' in next loop iteration
                else:
                    # Failed to launch game
                    if not self._check_and_reset_retry(step_name):
                        return  # Bot stopped due to max retries
                    logger.warning(f"Failed to launch game, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s...")
                    time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                    return
            else:
                # Game is running - set state to "running" and detect actual game state from screenshot
                self._update_game_state({
                    'running_state': 'running',
                    'current_state': 'running'
                })
                
                logger.info("Game is running - attempting to detect actual game state from screenshot...")
                
                # Take a screenshot for state detection
                screenshot = self._get_cached_screenshot()
                if screenshot is None:
                    screenshot = self.adb.take_screenshot()
                
                if screenshot is not None:
                    logger.info(f"Screenshot captured successfully (shape: {screenshot.shape})")
                    
                    # Store previous state before detection
                    previous_state = self.game_state['actual_game_state']
                    
                    # Detect actual game state based on specific element combinations
                    logger.info("Calling detect_game_state_from_screenshot()...")
                    detected_state = self.game_detector.detect_game_state_from_screenshot(
                        screenshot, 
                        debug=self.debug_detection
                    )
                    logger.info(f"Detection complete! Game running state: 'running', Detected state: '{detected_state}', Previous state: '{previous_state}'")
                    
                    # Update actual_game_state (restricted to: select_server, select_character, playing, unknown)
                    # detailed_game_state is independent and won't be reset automatically
                    valid_states = ['select_server', 'select_character', 'playing', 'unknown']
                    if detected_state not in valid_states:
                        # If detected state is not in valid list, default to 'unknown'
                        logger.warning(f"Invalid detected state '{detected_state}', defaulting to 'unknown'")
                        detected_state = 'unknown'
                    
                    # Update only actual_game_state, keep detailed_game_state unchanged
                    self._update_game_state({
                        'actual_game_state': detected_state
                    })
                    
                    logger.info(f"Current game state: '{self.game_state['actual_game_state']}'")
                    
                    # If playing state detected, check for agent quest button and tap it
                    # But only if detailed_game_state is not 'auto_questing'
                    if self.game_state['actual_game_state'] == 'playing':
                        # Take a fresh screenshot for player parameter detection to ensure accuracy
                        param_screenshot = self.adb.take_screenshot()
                        if param_screenshot is None:
                            logger.warning("Failed to take screenshot for player parameter detection")
                            param_screenshot = screenshot  # Fallback to original screenshot
                        
                        # Detect player parameters (blood bottle number and bag weight)
                        player_params = self.game_detector.detect_player_parameters(param_screenshot)
                        if player_params:
                            # Update current player parameters
                            if 'blood_bottle' in player_params:
                                self.player_parameters['blood_bottle'] = player_params['blood_bottle']
                            if 'bag_weight' in player_params:
                                self.player_parameters['bag_weight'] = player_params['bag_weight']
                            
                            # Update timestamp
                            self.player_parameters['last_updated'] = time.time()
                            
                            # Add to history (keep last 100 entries)
                            param_snapshot = {
                                'blood_bottle': self.player_parameters['blood_bottle'],
                                'bag_weight': self.player_parameters['bag_weight'],
                                'timestamp': self.player_parameters['last_updated']
                            }
                            self.player_parameters_history.append(param_snapshot)
                            if len(self.player_parameters_history) > 100:
                                self.player_parameters_history.pop(0)  # Remove oldest entry
                            
                            logger.info(f"Player parameters updated: {self.player_parameters}")
                            
                        
                        # Check blood bottle - only tap quest button if blood bottle is more than 100
                        blood_bottle = self.player_parameters.get('blood_bottle')
                        blood_bottle_sufficient = False
                        
                        if blood_bottle is None:
                            blood_bottle_sufficient = False
                        elif blood_bottle == '0':
                            blood_bottle_sufficient = False
                        else:
                            try:
                                blood_bottle_num = int(blood_bottle)
                                if blood_bottle_num > 100:
                                    blood_bottle_sufficient = True
                            except (ValueError, TypeError):
                                blood_bottle_sufficient = False
                        
                        # Only tap quest button if blood bottle is more than 100
                        if blood_bottle_sufficient:
                            if self.game_state['detailed_game_state'] != 'auto_questing':
                                logger.info("Playing state detected and detailed_game_state is not 'auto_questing' - checking for agent quest button...")
                                button_result = self.game_detector.detect_and_tap_agent_quest_button(screenshot)
                                if button_result == 'auto_questing':
                                    # Confirm button was tapped - set detailed_game_state to auto_questing
                                    logger.info("Confirm button tapped - setting detailed_game_state to 'auto_questing'")
                                    self._update_game_state({'detailed_game_state': 'auto_questing'})
                                elif button_result:
                                    logger.info("Quest button detected and tapped successfully")
                    # If already in auto_questing state, don't try to tap quest button
                    elif self.game_state['actual_game_state'] == 'auto_questing':
                        pass
                    # If select_server state detected, tap the screen
                    elif self.game_state['actual_game_state'] == 'select_server':
                        logger.info("Select server state detected - tapping screen...")
                        # Tap center of screen to proceed
                        if screenshot is not None:
                            screen_center_x = screenshot.shape[1] // 2
                            screen_center_y = screenshot.shape[0] // 2
                            if self.adb and hasattr(self.adb, 'tap'):
                                success = self.adb.tap(screen_center_x, screen_center_y)
                                if success:
                                    logger.info(f"Successfully tapped screen center at ({screen_center_x}, {screen_center_y})")
                                else:
                                    logger.warning(f"Failed to tap screen center at ({screen_center_x}, {screen_center_y})")
                            else:
                                logger.warning("ADB manager not available or tap method not found")
                    # If select_character state detected, tap enter button
                    elif self.game_state['actual_game_state'] == 'select_character':
                        logger.info("Select character state detected - detecting and tapping enter button...")
                        if screenshot is not None:
                            # Detect enter button
                            enter_result = self.game_detector.template_matcher.find_template(
                                screenshot,
                                "enter_button.png",
                                multi_scale=True,
                                confidence=0.7
                            )
                            if enter_result:
                                x, y, confidence = enter_result
                                logger.info(f"Enter button detected at ({x}, {y}) with confidence {confidence:.3f}")
                                if self.adb and hasattr(self.adb, 'tap'):
                                    success = self.adb.tap(int(x), int(y))
                                    if success:
                                        logger.info(f"Successfully tapped enter button at ({x}, {y})")
                                    else:
                                        logger.warning(f"Failed to tap enter button at ({x}, {y})")
                                else:
                                    logger.warning("ADB manager not available or tap method not found")
                            else:
                                logger.info("Enter button not detected in select_character state")
                    # If unknown state detected, check for common buttons and tap them
                    elif self.game_state['actual_game_state'] == 'unknown':
                        logger.info("Unknown state detected - checking for common buttons...")
                        button_tapped = self.game_detector.detect_and_tap_unknown_state_buttons(screenshot)
                        if button_tapped:
                            logger.info("Unknown state button detected and tapped successfully")
                    
                    # Get detailed detection results for debugging
                    if self.debug_detection:
                        detailed_results = self.game_detector.get_detection_results(screenshot)
                else:
                    logger.warning("Could not take screenshot for game state detection - setting state to 'unknown'")
                    self._update_game_state({'actual_game_state': 'unknown'})
            
            # State-based actions can use:
            # - self.game_state['running_state']: 'not_running' or 'running'
            # - self.game_state['actual_game_state']: 'select_server', 'select_character', 'playing', 'unknown', etc.
            # - self.game_state['current_state']: 'not_running' or 'running' (same as running_state)
            # - self.game_state['detailed_game_state']: 'auto_questing', etc.
            
            # Periodic garbage collection
            if not hasattr(self, '_loop_count'):
                self._loop_count = 0
            self._loop_count += 1
            
            if self._loop_count % 10 == 0:
                gc.collect(0)
                
        except Exception as e:
            logger.error(f"Error in game loop: {e}", exc_info=True)
            step_name = "game_loop_error"
            if not self._check_and_reset_retry(step_name):
                return
            time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
    
    def _update_game_state(self, updates: Dict[str, Any]):
        """
        Update game state dictionary and add to history
        
        Args:
            updates: Dictionary with state fields to update
        """
        # Update the game state dictionary
        self.game_state.update(updates)
        self.game_state['last_updated'] = time.time()
        
        # Add to history (keep last 100 entries)
        state_snapshot = self.game_state.copy()
        self.game_state_history.append(state_snapshot)
        if len(self.game_state_history) > 100:
            self.game_state_history.pop(0)  # Remove oldest entry
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get current game state information
        
        Returns:
            Dictionary with game state information:
            - 'current_state': 'not_running' or 'running'
            - 'running_state': 'not_running' or 'running'
            - 'actual_game_state': Actual game state from screenshot ('select_server', 'select_character', 'playing', 'unknown', etc.)
            - 'detailed_game_state': Detailed game state ('auto_questing', etc.)
            - 'last_updated': Timestamp of last update
        """
        return self.game_state.copy()
    
    def get_game_state_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get game state history
        
        Args:
            limit: Maximum number of history entries to return (None for all)
            
        Returns:
            List of state snapshots, each containing all game state fields
        """
        if limit is None:
            return self.game_state_history.copy()
        else:
            return self.game_state_history[-limit:].copy() if limit > 0 else []
    
    def get_player_parameters(self) -> Dict[str, Any]:
        """
        Get current player parameters
        
        Returns:
            Dictionary with current player parameters:
            - 'blood_bottle': str or None (blood bottle number)
            - 'bag_weight': str or None (bag weight)
            - 'last_updated': float or None (timestamp of last update)
        """
        return self.player_parameters.copy()
    
    def get_player_parameters_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get player parameters history
        
        Args:
            limit: Maximum number of history entries to return (None for all)
            
        Returns:
            List of parameter snapshots, each containing:
            - 'blood_bottle': str or None
            - 'bag_weight': str or None
            - 'timestamp': float
        """
        if limit is None:
            return self.player_parameters_history.copy()
        else:
            return self.player_parameters_history[-limit:].copy() if limit > 0 else []
    
    def _get_cached_screenshot(self, force_new: bool = False):
        """
        Get a cached screenshot or take a new one if cache is expired
        
        Args:
            force_new: Force taking a new screenshot even if cache is valid
            
        Returns:
            Screenshot numpy array or None
        """
        current_time = time.time()
        
        if (force_new or 
            self._last_screenshot is None or 
            (current_time - self._last_screenshot_time) > self._screenshot_cache_ttl):
            # Cache expired or forced, take new screenshot
            # Delete old screenshot before taking new one to free memory
            if self._last_screenshot is not None:
                del self._last_screenshot
                self._last_screenshot = None
                gc.collect(0)
            
            self._last_screenshot = self.adb.take_screenshot()
            self._last_screenshot_time = current_time
        
        return self._last_screenshot
    
    def _tap(self, x: int, y: int, duration: float = 0.1):
        """
        Execute a tap action at specified coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Tap duration in seconds
        """
        try:
            self.adb.tap(x, y)
            time.sleep(duration)
        except Exception as e:
            logger.error(f"Error tapping at ({x}, {y}): {e}")
    
    def _handle_general_merchant_selection(self):
        """
        Handle General Merchant selection after merchant button is tapped:
        1. Wait for merchant list to appear
        2. Detect "General Merchant" text from merchant_list_region
        3. If detected, tap it
        4. If not detected, swipe up to down in merchant_list_region
        5. Try to detect again
        6. If still not detected, tap merchant_button again
        """
        try:
            # Wait for merchant list to appear
            wait_time = random.uniform(1.0, 2.0)
            logger.info(f"Waiting {wait_time:.2f} seconds for merchant list to appear...")
            time.sleep(wait_time)
            
            # Take a screenshot
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot for General Merchant detection")
                return
            
            # Get merchant_list_region
            merchant_list_region = self.game_detector._get_region('merchant_list_region')
            if not merchant_list_region:
                logger.warning("merchant_list_region not configured - skipping General Merchant detection")
                return
            
            x1, y1, x2, y2 = merchant_list_region
            # Ensure coordinates are within screenshot bounds
            x1 = max(0, min(x1, screenshot.shape[1]))
            y1 = max(0, min(y1, screenshot.shape[0]))
            x2 = max(0, min(x2, screenshot.shape[1]))
            y2 = max(0, min(y2, screenshot.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid merchant_list_region bounds: ({x1}, {y1}, {x2}, {y2})")
                return
            
            # Extract region image
            region_image = screenshot[y1:y2, x1:x2]
            
            # Try to detect "General Merchant" text
            merchant_pos = self.game_detector._detect_general_merchant_text(region_image)
            
            if merchant_pos:
                # Convert relative coordinates to absolute coordinates
                abs_x = x1 + merchant_pos[0]
                abs_y = y1 + merchant_pos[1]
                logger.info(f"General Merchant detected at ({abs_x}, {abs_y}) - tapping...")
                if self.adb and hasattr(self.adb, 'tap'):
                    success = self.adb.tap(abs_x, abs_y)
                    if success:
                        logger.info("Successfully tapped General Merchant")
                        # Wait 30 seconds for character to arrive at merchant position
                        logger.info("Waiting 30 seconds for character to arrive at merchant position...")
                        time.sleep(30.0)
                        logger.info("Wait completed - character should be at merchant position")
                        return
                    else:
                        logger.warning(f"Failed to tap General Merchant at ({abs_x}, {abs_y})")
            else:
                logger.info("General Merchant not detected in merchant_list_region - swiping down...")
                
                # Swipe up to down in merchant_list_region
                # Swipe from top to bottom (scroll down)
                swipe_start_x = (x1 + x2) // 2
                swipe_start_y = y1 + (y2 - y1) // 4  # Start 1/4 from top
                swipe_end_x = swipe_start_x
                swipe_end_y = y2 - (y2 - y1) // 4  # End 1/4 from bottom
                
                logger.info(f"Swiping in merchant_list_region from ({swipe_start_x}, {swipe_start_y}) to ({swipe_end_x}, {swipe_end_y})")
                self._swipe(swipe_start_x, swipe_start_y, swipe_end_x, swipe_end_y, duration=300)
                
                # Wait for swipe to complete
                wait_time = random.uniform(0.5, 1.0)
                time.sleep(wait_time)
                
                # Take another screenshot and try again
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.warning("Failed to take screenshot after swipe")
                else:
                    region_image = screenshot[y1:y2, x1:x2]
                    merchant_pos = self.game_detector._detect_general_merchant_text(region_image)
                    
                    if merchant_pos:
                        # Convert relative coordinates to absolute coordinates
                        abs_x = x1 + merchant_pos[0]
                        abs_y = y1 + merchant_pos[1]
                        logger.info(f"General Merchant detected after swipe at ({abs_x}, {abs_y}) - tapping...")
                        if self.adb and hasattr(self.adb, 'tap'):
                            success = self.adb.tap(abs_x, abs_y)
                            if success:
                                logger.info("Successfully tapped General Merchant after swipe")
                                # Wait 30 seconds for character to arrive at merchant position
                                logger.info("Waiting 30 seconds for character to arrive at merchant position...")
                                time.sleep(30.0)
                                logger.info("Wait completed - character should be at merchant position")
                                return
                            else:
                                logger.warning(f"Failed to tap General Merchant at ({abs_x}, {abs_y})")
                    else:
                        logger.info("General Merchant still not detected after swipe - tapping merchant_button again...")
                        
                        # Tap merchant_button again using merchant_button_region
                        merchant_button_region = self.game_detector._get_region('merchant_button_region')
                        if merchant_button_region:
                            btn_x1, btn_y1, btn_x2, btn_y2 = merchant_button_region
                            tap_x = (btn_x1 + btn_x2) // 2
                            tap_y = (btn_y1 + btn_y2) // 2
                            logger.info(f"Tapping merchant_button_region at center: ({tap_x}, {tap_y})")
                            if self.adb and hasattr(self.adb, 'tap'):
                                success = self.adb.tap(tap_x, tap_y)
                                if success:
                                    logger.info("Successfully tapped merchant_button again")
                                else:
                                    logger.warning(f"Failed to tap merchant_button at ({tap_x}, {tap_y})")
                        else:
                            logger.warning("merchant_button_region not configured - cannot tap merchant button again")
                
        except Exception as e:
            logger.error(f"Error in General Merchant selection: {e}", exc_info=True)
    
    def _tap_merchant_and_select_general_merchant(self) -> bool:
        """
        Tap merchant button and detect/select General Merchant:
        1. Tap merchant_button_region
        2. Get new screenshot
        3. Detect "General Merchant" text in merchant_list_region
        4. If detected, tap it
        
        Returns:
            True if General Merchant was successfully selected, False otherwise
        """
        try:
            # Get merchant_button_region
            merchant_button_region = self.game_detector._get_region('merchant_button_region')
            if not merchant_button_region:
                logger.warning("merchant_button_region not configured - cannot tap merchant button")
                return False
            
            # Step 1: Take fresh screenshot before tapping merchant button
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot before tapping merchant button")
                return False
            
            # Step 2: Tap merchant_button_region
            btn_x1, btn_y1, btn_x2, btn_y2 = merchant_button_region
            tap_x = (btn_x1 + btn_x2) // 2
            tap_y = (btn_y1 + btn_y2) // 2
            logger.info(f"Tapping merchant_button_region at center: ({tap_x}, {tap_y})")
            
            if self.adb and hasattr(self.adb, 'tap'):
                success = self.adb.tap(tap_x, tap_y)
                if not success:
                    logger.warning(f"Failed to tap merchant_button_region at ({tap_x}, {tap_y})")
                    return False
            else:
                logger.warning("ADB manager not available or tap method not found")
                return False
            
            # Wait for merchant list to appear
            wait_time = random.uniform(1.0, 2.0)
            logger.info(f"Waiting {wait_time:.2f} seconds for merchant list to appear...")
            time.sleep(wait_time)
            
            # Step 3: Take fresh screenshot before detecting General Merchant
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot for General Merchant detection")
                return False
            
            # Step 4: Detect "General Merchant" text in merchant_list_region
            merchant_list_region = self.game_detector._get_region('merchant_list_region')
            if not merchant_list_region:
                logger.warning("merchant_list_region not configured - cannot detect General Merchant")
                return False
            
            x1, y1, x2, y2 = merchant_list_region
            # Ensure coordinates are within screenshot bounds
            x1 = max(0, min(x1, screenshot.shape[1]))
            y1 = max(0, min(y1, screenshot.shape[0]))
            x2 = max(0, min(x2, screenshot.shape[1]))
            y2 = max(0, min(y2, screenshot.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid merchant_list_region bounds: ({x1}, {y1}, {x2}, {y2})")
                return False
            
            # Extract region image
            region_image = screenshot[y1:y2, x1:x2]
            
            # Debug: Log region bounds and image size
            logger.debug(f"merchant_list_region bounds: ({x1}, {y1}, {x2}, {y2}), size: {region_image.shape}")
            
            # Try to detect "General Merchant" text
            logger.info("Attempting to detect 'General Merchant' text in merchant_list_region...")
            merchant_pos = self.game_detector._detect_general_merchant_text(region_image)
            
            # Step 5: If detected, take fresh screenshot and tap it
            if merchant_pos:
                # Take fresh screenshot before tapping General
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.warning("Failed to take screenshot before tapping General Merchant")
                    return False
                
                # Re-extract region from fresh screenshot
                x1, y1, x2, y2 = merchant_list_region
                x1 = max(0, min(x1, screenshot.shape[1]))
                y1 = max(0, min(y1, screenshot.shape[0]))
                x2 = max(0, min(x2, screenshot.shape[1]))
                y2 = max(0, min(y2, screenshot.shape[0]))
                region_image = screenshot[y1:y2, x1:x2]
                
                # Re-detect to get fresh coordinates
                merchant_pos = self.game_detector._detect_general_merchant_text(region_image)
                if not merchant_pos:
                    logger.warning("General Merchant position changed - retrying detection")
                    return False
                
                # Convert relative coordinates to absolute coordinates
                abs_x = x1 + merchant_pos[0]
                abs_y = y1 + merchant_pos[1]
                logger.info(f"General Merchant detected at ({abs_x}, {abs_y}) - tapping...")
                if self.adb and hasattr(self.adb, 'tap'):
                    success = self.adb.tap(abs_x, abs_y)
                    if success:
                        logger.info("Successfully tapped General Merchant")
                        # Wait 30 seconds for character to arrive at merchant position
                        logger.info("Waiting 30 seconds for character to arrive at merchant position...")
                        time.sleep(30.0)
                        logger.info("Wait completed - character should be at merchant position")
                        return True
                    else:
                        logger.warning(f"Failed to tap General Merchant at ({abs_x}, {abs_y})")
                        return False
                else:
                    logger.warning("ADB manager not available or tap method not found")
                    return False
            else:
                logger.info("General Merchant not detected in merchant_list_region")
                return False
                
        except Exception as e:
            logger.error(f"Error tapping merchant and selecting General Merchant: {e}", exc_info=True)
            return False
    
    def _handle_complete_purchasing_flow(self):
        """
        Complete purchasing flow:
        1. Tap merchant button and select General Merchant
        2. Wait for shop to open
        3. Detect and buy blood bottles
        4. Close merchant window
        5. Reset detailed_game_state
        """
        try:
            # Check if we're already in the shop (health item might already be selected)
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot - will retry in next cycle")
                return
            
            # Check if health item is already selected by checking if we're in the quantity selection screen
            # (If purchase_number_100_region exists and contains "100", it means health was already selected)
            purchase_number_100_region = self.game_detector._get_region('purchase_number_100_region')
            health_already_selected = False
            
            if purchase_number_100_region:
                # Try to detect number 100 in the region - if found, we're already in quantity selection screen
                number_100_pos = self._detect_number_100(screenshot, purchase_number_100_region)
                if number_100_pos:
                    health_already_selected = True
                    logger.info("Already in purchase quantity selection screen - health item already selected, skipping merchant and health selection")
            
            # Step 1: Tap merchant button and select General Merchant (only if not already in shop)
            if not health_already_selected:
                logger.info("Step 1: Tapping merchant button and selecting General Merchant...")
                merchant_selected = self._tap_merchant_and_select_general_merchant()
                
                if not merchant_selected:
                    logger.warning("Failed to select General Merchant - will retry in next cycle")
                    return
                
                # Step 2: Wait for shop to open
                wait_time = random.uniform(1.5, 2.5)
                logger.info(f"Step 2: Waiting {wait_time:.2f} seconds for shop to open...")
                time.sleep(wait_time)
                
                # Step 3: Detect and select "health" item
                logger.info("Step 3: Detecting and selecting 'health' item...")
                health_selected = self._detect_and_select_health_item()
                
                if not health_selected:
                    logger.warning("Failed to select 'health' item - will retry in next cycle")
                    return
            else:
                logger.info("Step 1-3: Skipping merchant and health selection (already completed)")
            
            # Step 4: Select purchase quantity based on weight
            logger.info("Step 4: Selecting purchase quantity...")
            purchase_success = self._select_purchase_quantity()
            
            if purchase_success:
                logger.info("Successfully selected purchase quantity")
            else:
                logger.warning("Failed to select purchase quantity - will retry in next cycle")
                return
            
            # Step 5: Confirm purchase
            logger.info("Step 5: Confirming purchase...")
            confirm_success = self._confirm_purchase()
            
            if not confirm_success:
                logger.warning("Failed to confirm purchase - will retry in next cycle")
                return
            
            # Step 6: Close merchant window
            logger.info("Step 6: Closing merchant window...")
            self._close_merchant_window()
            
            # Step 7: Reset detailed_game_state
            logger.info("Step 7: Resetting detailed_game_state...")
            self._update_game_state({'detailed_game_state': 'unknown'})
            logger.info("Purchasing flow completed successfully")
            
        except Exception as e:
            logger.error(f"Error in complete purchasing flow: {e}", exc_info=True)
    
    def _detect_and_select_health_item(self) -> bool:
        """
        Detect "health" text in purchase_list_region and tap it.
        If not detected, swipe to search for it like a human would.
        
        Returns:
            True if "health" item was successfully selected, False otherwise
        """
        try:
            purchase_list_region = self.game_detector._get_region('purchase_list_region')
            if not purchase_list_region:
                logger.warning("purchase_list_region not configured - cannot detect health item")
                return False
            
            max_swipe_attempts = 5  # Maximum number of swipe attempts
            swipe_attempt = 0
            
            while swipe_attempt < max_swipe_attempts:
                # Take screenshot
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.warning("Failed to take screenshot for health item detection")
                    return False
                
                x1, y1, x2, y2 = purchase_list_region
                # Ensure coordinates are within screenshot bounds
                x1 = max(0, min(x1, screenshot.shape[1]))
                y1 = max(0, min(y1, screenshot.shape[0]))
                x2 = max(0, min(x2, screenshot.shape[1]))
                y2 = max(0, min(y2, screenshot.shape[0]))
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid purchase_list_region bounds: ({x1}, {y1}, {x2}, {y2})")
                    return False
                
                # Extract region image
                region_image = screenshot[y1:y2, x1:x2]
                
                # Try to detect "health" text using OCR
                health_pos = self._detect_health_text(region_image)
                
                if health_pos:
                    # Take fresh screenshot before tapping health item
                    screenshot = self.adb.take_screenshot()
                    if screenshot is None:
                        logger.warning("Failed to take screenshot before tapping health item")
                        return False
                    
                    # Re-extract region from fresh screenshot
                    x1, y1, x2, y2 = purchase_list_region
                    x1 = max(0, min(x1, screenshot.shape[1]))
                    y1 = max(0, min(y1, screenshot.shape[0]))
                    x2 = max(0, min(x2, screenshot.shape[1]))
                    y2 = max(0, min(y2, screenshot.shape[0]))
                    region_image = screenshot[y1:y2, x1:x2]
                    
                    # Re-detect to get fresh coordinates
                    health_pos = self._detect_health_text(region_image)
                    if not health_pos:
                        logger.warning("Health item position changed - retrying detection")
                        continue
                    
                    # Convert relative coordinates to absolute coordinates
                    abs_x = x1 + health_pos[0]
                    abs_y = y1 + health_pos[1]
                    logger.info(f"Health item detected at ({abs_x}, {abs_y}) - tapping...")
                    
                    if self.adb and hasattr(self.adb, 'tap'):
                        success = self.adb.tap(abs_x, abs_y)
                        if success:
                            logger.info("Successfully tapped health item")
                            # Wait for health item screen to appear
                            logger.info("Waiting for health item screen to appear...")
                            health_appeared = self._wait_for_health_screen()
                            if health_appeared:
                                logger.info("Health item screen appeared - proceeding to quantity selection")
                                return True
                            else:
                                logger.warning("Health item screen did not appear after tapping - may proceed anyway")
                                # Still return True to continue, but log warning
                                return True
                        else:
                            logger.warning(f"Failed to tap health item at ({abs_x}, {abs_y})")
                    else:
                        logger.warning("ADB manager not available or tap method not found")
                        return False
                else:
                    # Health not found, try swiping
                    if swipe_attempt < max_swipe_attempts - 1:
                        logger.info(f"Health item not detected - swiping to search (attempt {swipe_attempt + 1}/{max_swipe_attempts - 1})...")
                        # Swipe down to scroll (like a human would)
                        swipe_start_x = (x1 + x2) // 2
                        swipe_start_y = y1 + (y2 - y1) // 4  # Start 1/4 from top
                        swipe_end_x = swipe_start_x
                        swipe_end_y = y2 - (y2 - y1) // 4  # End 1/4 from bottom
                        
                        # Randomize swipe slightly for human-like behavior
                        swipe_start_x += random.randint(-20, 20)
                        swipe_end_x += random.randint(-20, 20)
                        
                        logger.info(f"Swiping in purchase_list_region from ({swipe_start_x}, {swipe_start_y}) to ({swipe_end_x}, {swipe_end_y})")
                        self._swipe(swipe_start_x, swipe_start_y, swipe_end_x, swipe_end_y, duration=random.randint(300, 500))
                        
                        # Wait for swipe to complete
                        wait_time = random.uniform(0.5, 1.0)
                        time.sleep(wait_time)
                        
                        swipe_attempt += 1
                    else:
                        logger.warning("Health item not found after maximum swipe attempts")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting and selecting health item: {e}", exc_info=True)
            return False
    
    def _detect_health_text(self, region_image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect "health" text in a region using OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            Tuple of (x, y) center coordinates of the detected text, or None if not found
        """
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                logger.warning("Tesseract OCR reader not available for health detection")
                return None
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Look for "health" text
                search_text = "health"
                search_variations = ["health", "heal", "hea"]  # Partial matches
                
                # Strategy 1: Check if search text appears in any single OCR result
                for idx, (bbox, text, confidence) in enumerate(results):
                    text_lower = text.lower().strip()
                    for variation in search_variations:
                        if variation in text_lower:
                            # Calculate center of bounding box
                            if len(bbox) >= 4:
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                                center_x = int(sum(x_coords) / len(x_coords))
                                center_y = int(sum(y_coords) / len(y_coords))
                                logger.info(f"Detected 'health' text at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                                return (center_x, center_y)
                
                # Strategy 2: Check for exact match or starts with "health"
                for idx, (bbox, text, confidence) in enumerate(results):
                    text_lower = text.lower().strip()
                    if text_lower == "health" or text_lower.startswith("health"):
                        # Calculate center of bounding box
                        if len(bbox) >= 4:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            center_x = int(sum(x_coords) / len(x_coords))
                            center_y = int(sum(y_coords) / len(y_coords))
                            logger.info(f"Detected 'health' text (exact/prefix match) at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                            return (center_x, center_y)
                
                return None
                
            finally:
                if results is not None:
                    del results
                    
        except Exception as e:
            logger.error(f"Error detecting health text: {e}", exc_info=True)
            return None
    
    def _wait_for_health_screen(self, max_wait_time: float = 5.0, check_interval: float = 0.5) -> bool:
        """
        Wait until health item screen appears after tapping health.
        Checks for presence of purchase_number_100_region or estimated_weight_region.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            check_interval: Interval between checks in seconds
            
        Returns:
            True if health screen appeared, False if timeout
        """
        try:
            start_time = time.time()
            check_count = 0
            max_checks = int(max_wait_time / check_interval)
            
            purchase_number_100_region = self.game_detector._get_region('purchase_number_100_region')
            estimated_weight_region = self.game_detector._get_region('estimated_weight_region')
            
            if not purchase_number_100_region and not estimated_weight_region:
                logger.warning("Neither purchase_number_100_region nor estimated_weight_region configured - cannot verify health screen")
                # Wait a default time and return True
                wait_time = random.uniform(1.5, 2.5)
                time.sleep(wait_time)
                return True
            
            while check_count < max_checks:
                elapsed = time.time() - start_time
                if elapsed >= max_wait_time:
                    logger.warning(f"Timeout waiting for health screen after {max_wait_time:.1f} seconds")
                    return False
                
                # Take screenshot
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.warning("Failed to take screenshot while waiting for health screen")
                    time.sleep(check_interval)
                    check_count += 1
                    continue
                
                # Check if purchase_number_100_region contains "100" (indicates quantity selection screen is ready)
                if purchase_number_100_region:
                    number_100_pos = self._detect_number_100(screenshot, purchase_number_100_region)
                    if number_100_pos:
                        logger.info(f"Health screen appeared - detected number 100 after {elapsed:.2f} seconds")
                        return True
                
                # Also check if estimated_weight_region contains a number (indicates weight is displayed)
                if estimated_weight_region:
                    estimated_weight = self._detect_estimated_weight(screenshot, estimated_weight_region)
                    if estimated_weight is not None:
                        logger.info(f"Health screen appeared - detected estimated weight ({estimated_weight}) after {elapsed:.2f} seconds")
                        return True
                
                # Wait before next check
                time.sleep(check_interval)
                check_count += 1
            
            logger.warning(f"Health screen did not appear after {max_wait_time:.1f} seconds of checking")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for health screen: {e}", exc_info=True)
            # On error, wait a default time and return True to continue
            wait_time = random.uniform(1.5, 2.5)
            time.sleep(wait_time)
            return True
    
    def _select_purchase_quantity(self) -> bool:
        """
        Select purchase quantity by tapping number 100 repeatedly.
        Stop when estimated weight exceeds 60.
        
        Returns:
            True if quantity selection completed successfully, False otherwise
        """
        try:
            purchase_number_100_region = self.game_detector._get_region('purchase_number_100_region')
            estimated_weight_region = self.game_detector._get_region('estimated_weight_region')
            
            if not purchase_number_100_region:
                logger.warning("purchase_number_100_region not configured - cannot select quantity")
                return False
            
            if not estimated_weight_region:
                logger.warning("estimated_weight_region not configured - cannot check weight limit")
                return False
            
            # Take initial screenshot and verify we're in the purchase screen
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot for purchase quantity selection")
                return False
            
            # First, try to detect number 100 to confirm we're in the right screen
            logger.info("Checking if purchase quantity screen is ready...")
            number_100_pos = self._detect_number_100(screenshot, purchase_number_100_region)
            if not number_100_pos:
                logger.warning("Number 100 not detected - purchase screen may not be ready yet")
                # Wait a bit and try once more
                wait_time = random.uniform(1.0, 2.0)
                logger.info(f"Waiting {wait_time:.2f} seconds and retrying...")
                time.sleep(wait_time)
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    return False
                number_100_pos = self._detect_number_100(screenshot, purchase_number_100_region)
                if not number_100_pos:
                    logger.warning("Number 100 still not detected after wait - cannot proceed with quantity selection")
                    return False
            
            max_taps = 20  # Maximum number of taps to prevent infinite loop
            tap_count = 0
            
            while tap_count < max_taps:
                # Check current estimated weight
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.warning("Failed to take screenshot for weight check")
                    return False
                
                estimated_weight = self._detect_estimated_weight(screenshot, estimated_weight_region)
                
                if estimated_weight is not None:
                    logger.info(f"Current estimated weight: {estimated_weight}")
                    
                    if estimated_weight > 60:
                        logger.info(f"Estimated weight ({estimated_weight}) exceeds 60 - stopping quantity selection")
                        return True
                
                # Take fresh screenshot before detecting and tapping number 100
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.warning("Failed to take screenshot before tapping number 100")
                    return False
                
                # Detect and tap number 100
                number_100_pos = self._detect_number_100(screenshot, purchase_number_100_region)
                
                if number_100_pos:
                    abs_x, abs_y = number_100_pos
                    logger.info(f"Number 100 detected at ({abs_x}, {abs_y}) - tapping...")
                    
                    if self.adb and hasattr(self.adb, 'tap'):
                        success = self.adb.tap(abs_x, abs_y)
                        if success:
                            logger.info(f"Successfully tapped number 100 (tap {tap_count + 1}/{max_taps})")
                            tap_count += 1
                            
                            # Wait a bit before next tap (human-like delay)
                            wait_time = random.uniform(0.3, 0.6)
                            time.sleep(wait_time)
                        else:
                            logger.warning(f"Failed to tap number 100 at ({abs_x}, {abs_y})")
                            return False
                    else:
                        logger.warning("ADB manager not available or tap method not found")
                        return False
                else:
                    logger.warning("Number 100 not detected in purchase_number_100_region")
                    return False
            
            logger.warning(f"Reached maximum tap count ({max_taps}) - stopping quantity selection")
            return True
            
        except Exception as e:
            logger.error(f"Error selecting purchase quantity: {e}", exc_info=True)
            return False
    
    def _detect_number_100(self, screenshot: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """
        Detect number 100 in a region using OCR
        
        Args:
            screenshot: Full screenshot
            region: Region coordinates (x1, y1, x2, y2)
            
        Returns:
            Tuple of (x, y) center coordinates of the detected number, or None if not found
        """
        try:
            x1, y1, x2, y2 = region
            # Ensure coordinates are within screenshot bounds
            x1 = max(0, min(x1, screenshot.shape[1]))
            y1 = max(0, min(y1, screenshot.shape[0]))
            x2 = max(0, min(x2, screenshot.shape[1]))
            y2 = max(0, min(y2, screenshot.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract region image
            region_image = screenshot[y1:y2, x1:x2]
            
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return None
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Look for "100" text
                for idx, (bbox, text, confidence) in enumerate(results):
                    text_clean = text.strip()
                    # Check for "100" or just "100" as digits
                    if "100" in text_clean or text_clean == "100":
                        # Calculate center of bounding box
                        if len(bbox) >= 4:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            center_x = int(sum(x_coords) / len(x_coords))
                            center_y = int(sum(y_coords) / len(y_coords))
                            # Convert to absolute coordinates
                            abs_x = x1 + center_x
                            abs_y = y1 + center_y
                            logger.debug(f"Detected '100' at ({abs_x}, {abs_y}) with confidence {confidence:.3f}")
                            return (abs_x, abs_y)
                
                return None
                
            finally:
                if results is not None:
                    del results
                    
        except Exception as e:
            logger.error(f"Error detecting number 100: {e}", exc_info=True)
            return None
    
    def _detect_estimated_weight(self, screenshot: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Detect estimated weight number in a region using OCR
        
        Args:
            screenshot: Full screenshot
            region: Region coordinates (x1, y1, x2, y2)
            
        Returns:
            Estimated weight as float, or None if not detected
        """
        try:
            x1, y1, x2, y2 = region
            # Ensure coordinates are within screenshot bounds
            x1 = max(0, min(x1, screenshot.shape[1]))
            y1 = max(0, min(y1, screenshot.shape[0]))
            x2 = max(0, min(x2, screenshot.shape[1]))
            y2 = max(0, min(y2, screenshot.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract region image
            region_image = screenshot[y1:y2, x1:x2]
            
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return None
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Extract text and look for numbers
                for idx, (bbox, text, confidence) in enumerate(results):
                    # Extract numbers from text
                    numbers = re.findall(r'\d+\.?\d*', text)
                    if numbers:
                        try:
                            # Try to parse the first number found
                            weight = float(numbers[0])
                            logger.debug(f"Detected estimated weight: {weight}")
                            return weight
                        except ValueError:
                            continue
                
                return None
                
            finally:
                if results is not None:
                    del results
                    
        except Exception as e:
            logger.error(f"Error detecting estimated weight: {e}", exc_info=True)
            return None
    
    def _confirm_purchase(self) -> bool:
        """
        Detect and tap confirm button
        
        Returns:
            True if confirm button was successfully tapped, False otherwise
        """
        try:
            # Take fresh screenshot before detecting confirm button
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot for confirm button detection")
                return False
            
            # Try to detect confirm button using image template
            confirm_button_template = "confirm_button.png"
            result = self.game_detector.template_matcher.find_template(
                screenshot,
                confirm_button_template,
                multi_scale=True,
                confidence=0.6
            )
            
            if result:
                x, y, confidence = result
                logger.info(f"Confirm button detected at ({x}, {y}) with confidence {confidence:.3f}")
                
                # Take fresh screenshot before tapping confirm button
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.warning("Failed to take screenshot before tapping confirm button")
                    return False
                
                # Re-detect to get fresh coordinates
                result = self.game_detector.template_matcher.find_template(
                    screenshot,
                    confirm_button_template,
                    multi_scale=True,
                    confidence=0.6
                )
                if not result:
                    logger.warning("Confirm button position changed - retrying detection")
                    return False
                
                x, y, confidence = result
                
                if self.adb and hasattr(self.adb, 'tap'):
                    success = self.adb.tap(int(x), int(y))
                    if success:
                        logger.info("Successfully tapped confirm button")
                        return True
                    else:
                        logger.warning(f"Failed to tap confirm button at ({x}, {y})")
                        return False
            else:
                # Try to use confirm_button_region if configured
                confirm_button_region = self.game_detector._get_region('confirm_button_region')
                if confirm_button_region:
                    x1, y1, x2, y2 = confirm_button_region
                    tap_x = (x1 + x2) // 2
                    tap_y = (y1 + y2) // 2
                    
                    if self.adb and hasattr(self.adb, 'tap'):
                        success = self.adb.tap(tap_x, tap_y)
                        if success:
                            logger.info("Successfully tapped confirm button using region")
                            return True
                        else:
                            logger.warning(f"Failed to tap confirm button at ({tap_x}, {tap_y})")
                            return False
                else:
                    logger.warning("Confirm button not detected and confirm_button_region not configured")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error confirming purchase: {e}", exc_info=True)
            return False
    
    def _buy_blood_bottles(self) -> bool:
        """
        Detect and buy blood bottles from merchant shop
        
        Returns:
            True if blood bottles were successfully purchased, False otherwise
        """
        try:
            # Take screenshot
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot for blood bottle purchase")
                return False
            
            # Try to detect blood bottle item in shop
            # Option 1: Try to detect using image template
            blood_bottle_template = "blood_bottle_item.png"
            result = self.game_detector.template_matcher.find_template(
                screenshot,
                blood_bottle_template,
                multi_scale=True,
                confidence=0.6
            )
            
            if result:
                x, y, confidence = result
                logger.info(f"Blood bottle item detected at ({x}, {y}) with confidence {confidence:.3f}")
                
                # Tap on the blood bottle item
                if self.adb and hasattr(self.adb, 'tap'):
                    success = self.adb.tap(int(x), int(y))
                    if not success:
                        logger.warning(f"Failed to tap blood bottle item at ({x}, {y})")
                        return False
                    
                    # Wait for buy button to appear
                    wait_time = random.uniform(0.5, 1.0)
                    time.sleep(wait_time)
                    
                    # Try to detect and tap buy button
                    buy_button_template = "buy_button.png"
                    screenshot = self.adb.take_screenshot()
                    if screenshot is not None:
                        buy_result = self.game_detector.template_matcher.find_template(
                            screenshot,
                            buy_button_template,
                            multi_scale=True,
                            confidence=0.6
                        )
                        
                        if buy_result:
                            buy_x, buy_y, buy_confidence = buy_result
                            logger.info(f"Buy button detected at ({buy_x}, {buy_y}) with confidence {buy_confidence:.3f}")
                            
                            if self.adb and hasattr(self.adb, 'tap'):
                                success = self.adb.tap(int(buy_x), int(buy_y))
                                if success:
                                    logger.info("Successfully tapped buy button")
                                    
                                    # Wait for confirmation dialog
                                    wait_time = random.uniform(0.5, 1.0)
                                    time.sleep(wait_time)
                                    
                                    # Try to detect and tap confirm button
                                    confirm_button_template = "confirm_button.png"
                                    screenshot = self.adb.take_screenshot()
                                    if screenshot is not None:
                                        confirm_result = self.game_detector.template_matcher.find_template(
                                            screenshot,
                                            confirm_button_template,
                                            multi_scale=True,
                                            confidence=0.6
                                        )
                                        
                                        if confirm_result:
                                            confirm_x, confirm_y, confirm_confidence = confirm_result
                                            logger.info(f"Confirm button detected at ({confirm_x}, {confirm_y}) with confidence {confirm_confidence:.3f}")
                                            
                                            if self.adb and hasattr(self.adb, 'tap'):
                                                success = self.adb.tap(int(confirm_x), int(confirm_y))
                                                if success:
                                                    logger.info("Successfully confirmed purchase")
                                                    return True
                                                else:
                                                    logger.warning(f"Failed to tap confirm button at ({confirm_x}, {confirm_y})")
                                        else:
                                            logger.info("Confirm button not detected - purchase may have completed automatically")
                                            return True
                        else:
                            logger.info("Buy button not detected - trying alternative method")
                    else:
                        logger.warning("Failed to take screenshot for buy button detection")
                else:
                    logger.warning("ADB manager not available or tap method not found")
                    return False
            else:
                # Option 2: Try to detect using OCR text detection
                logger.info("Blood bottle item not detected via image - trying OCR text detection...")
                
                # Get shop item region if configured
                shop_item_region = self.game_detector._get_region('shop_item_region')
                if shop_item_region:
                    x1, y1, x2, y2 = shop_item_region
                    x1 = max(0, min(x1, screenshot.shape[1]))
                    y1 = max(0, min(y1, screenshot.shape[0]))
                    x2 = max(0, min(x2, screenshot.shape[1]))
                    y2 = max(0, min(y2, screenshot.shape[0]))
                    
                    if x2 > x1 and y2 > y1:
                        region_image = screenshot[y1:y2, x1:x2]
                        text = self.game_detector._extract_text_from_region(region_image)
                        
                        if text and ('blood' in text.lower() or 'bottle' in text.lower() or 'potion' in text.lower()):
                            logger.info(f"Detected blood bottle text in shop: {text}")
                            # Tap center of shop item region
                            tap_x = (x1 + x2) // 2
                            tap_y = (y1 + y2) // 2
                            
                            if self.adb and hasattr(self.adb, 'tap'):
                                success = self.adb.tap(tap_x, tap_y)
                                if success:
                                    logger.info(f"Tapped blood bottle item at ({tap_x}, {tap_y})")
                                    
                                    # Wait and try to tap buy button
                                    wait_time = random.uniform(0.5, 1.0)
                                    time.sleep(wait_time)
                                    
                                    # Try to find and tap buy button
                                    buy_button_region = self.game_detector._get_region('buy_button_region')
                                    if buy_button_region:
                                        btn_x1, btn_y1, btn_x2, btn_y2 = buy_button_region
                                        buy_tap_x = (btn_x1 + btn_x2) // 2
                                        buy_tap_y = (btn_y1 + btn_y2) // 2
                                        
                                        screenshot = self.adb.take_screenshot()
                                        if screenshot is not None:
                                            success = self.adb.tap(buy_tap_x, buy_tap_y)
                                            if success:
                                                logger.info("Tapped buy button")
                                                
                                                # Wait for confirmation
                                                wait_time = random.uniform(0.5, 1.0)
                                                time.sleep(wait_time)
                                                
                                                # Try to tap confirm button
                                                confirm_button_region = self.game_detector._get_region('confirm_button_region')
                                                if confirm_button_region:
                                                    conf_x1, conf_y1, conf_x2, conf_y2 = confirm_button_region
                                                    conf_tap_x = (conf_x1 + conf_x2) // 2
                                                    conf_tap_y = (conf_y1 + conf_y2) // 2
                                                    
                                                    success = self.adb.tap(conf_tap_x, conf_tap_y)
                                                    if success:
                                                        logger.info("Tapped confirm button")
                                                        return True
                                    else:
                                        logger.warning("buy_button_region not configured")
                else:
                    logger.warning("shop_item_region not configured for OCR detection")
            
            logger.warning("Failed to purchase blood bottles")
            return False
            
        except Exception as e:
            logger.error(f"Error buying blood bottles: {e}", exc_info=True)
            return False
    
    def _close_merchant_window(self):
        """
        Close the merchant window by tapping close button or back button
        """
        try:
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                logger.warning("Failed to take screenshot for closing merchant window")
                return
            
            # Try to detect close button
            close_button_template = "close_button.png"
            result = self.game_detector.template_matcher.find_template(
                screenshot,
                close_button_template,
                multi_scale=True,
                confidence=0.6
            )
            
            if result:
                x, y, confidence = result
                logger.info(f"Close button detected at ({x}, {y}) with confidence {confidence:.3f}")
                
                if self.adb and hasattr(self.adb, 'tap'):
                    success = self.adb.tap(int(x), int(y))
                    if success:
                        logger.info("Successfully closed merchant window")
                        return
            else:
                # Try to use close_button_region if configured
                close_button_region = self.game_detector._get_region('close_button_region')
                if close_button_region:
                    x1, y1, x2, y2 = close_button_region
                    tap_x = (x1 + x2) // 2
                    tap_y = (y1 + y2) // 2
                    
                    if self.adb and hasattr(self.adb, 'tap'):
                        success = self.adb.tap(tap_x, tap_y)
                        if success:
                            logger.info("Successfully closed merchant window using region")
                            return
                else:
                    # Fallback: try back button
                    back_button_template = "back_button_1.png"
                    result = self.game_detector.template_matcher.find_template(
                        screenshot,
                        back_button_template,
                        multi_scale=True,
                        confidence=0.6
                    )
                    
                    if result:
                        x, y, confidence = result
                        logger.info(f"Back button detected at ({x}, {y}) - using as close button")
                        
                        if self.adb and hasattr(self.adb, 'tap'):
                            success = self.adb.tap(int(x), int(y))
                            if success:
                                logger.info("Successfully closed merchant window using back button")
                                return
                    
                    logger.warning("Could not find close button - merchant window may remain open")
                    
        except Exception as e:
            logger.error(f"Error closing merchant window: {e}", exc_info=True)
    
    def _swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """
        Execute a swipe action
        
        Args:
            x1: Start X coordinate
            y1: Start Y coordinate
            x2: End X coordinate
            y2: End Y coordinate
            duration: Swipe duration in milliseconds
        """
        try:
            success, output = self.adb.execute_adb_command([
                'shell', 'input', 'swipe', 
                str(x1), str(y1), str(x2), str(y2), str(duration)
            ])
            
            if success:
                pass
            else:
                logger.warning(f"Failed to swipe: {output}")
        except Exception as e:
            logger.error(f"Error swiping: {e}")
