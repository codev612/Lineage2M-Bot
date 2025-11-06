"""
Game Automation Module - Implements actual gameplay scenarios for Lineage 2M
Handles game actions, state detection, and automated gameplay strategies
"""

import time
import gc
from typing import Dict, Optional, Tuple
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
        
        # Game state tracking
        self.current_state = 'unknown'
        self.game_running_state = 'unknown'  # 'not_running' or 'running'
        self.actual_game_state = 'unknown'  # Actual game state from screenshot: 'select_server', 'select_character', 'playing', etc.
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
        
    def start(self):
        """Start the game automation"""
        if self.running:
            logger.warning("Game automation already running")
            return
        
        logger.info("Starting game automation...")
        self.running = True
        
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
                self.game_running_state = 'not_running'
                self.current_state = 'not_running'
                self.actual_game_state = 'unknown'
                
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
                self.game_running_state = 'running'
                self.current_state = 'running'
                
                logger.info("Game is running - attempting to detect actual game state from screenshot...")
                
                # Take a screenshot for state detection
                logger.debug("Taking screenshot for game state detection...")
                screenshot = self._get_cached_screenshot()
                if screenshot is None:
                    logger.debug("Cached screenshot not available, taking new screenshot...")
                    screenshot = self.adb.take_screenshot()
                
                if screenshot is not None:
                    logger.info(f"Screenshot captured successfully (shape: {screenshot.shape})")
                    # Detect actual game state based on specific element combinations
                    logger.info("Calling detect_game_state_from_screenshot()...")
                    self.actual_game_state = self.game_detector.detect_game_state_from_screenshot(
                        screenshot, 
                        debug=self.debug_detection
                    )
                    logger.info(f"Detection complete! Game running state: 'running', Actual game state detected: '{self.actual_game_state}'")
                    
                    # Get detailed detection results for debugging
                    if self.debug_detection:
                        detailed_results = self.game_detector.get_detection_results(screenshot)
                        logger.info(f"Detailed detection results: {detailed_results}")
                else:
                    logger.warning("Could not take screenshot for game state detection - setting state to 'unknown'")
                    self.actual_game_state = 'unknown'
            
            # State-based actions can use:
            # - self.game_running_state: 'not_running' or 'running'
            # - self.actual_game_state: 'select_server', 'select_character', 'playing', 'unknown', etc.
            # - self.current_state: 'not_running' or 'running' (same as game_running_state)
            
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
    
    def get_game_state(self) -> Dict[str, str]:
        """
        Get current game state information
        
        Returns:
            Dictionary with game state information:
            - 'running_state': 'not_running' or 'running'
            - 'current_state': 'not_running' or 'running' (same as running_state)
            - 'actual_game_state': Actual game state from screenshot ('select_server', 'select_character', 'playing', 'unknown', etc.)
        """
        return {
            'running_state': self.game_running_state,
            'current_state': self.current_state,
            'actual_game_state': self.actual_game_state
        }
    
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
            logger.debug(f"Tapped at ({x}, {y})")
            time.sleep(duration)
        except Exception as e:
            logger.error(f"Error tapping at ({x}, {y}): {e}")
    
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
                logger.debug(f"Swiped from ({x1}, {y1}) to ({x2}, {y2})")
            else:
                logger.warning(f"Failed to swipe: {output}")
        except Exception as e:
            logger.error(f"Error swiping: {e}")
