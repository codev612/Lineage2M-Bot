"""
Game Automation Module - Implements actual gameplay scenarios for Lineage 2M
Handles game actions, state detection, and automated gameplay strategies
"""

import time
import random
import gc
from typing import Dict, List, Optional, Tuple, Any
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
DEFAULT_SCREEN_TRANSITION_TIME = 30.0  # Default time to wait for screen transitions (seconds)
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
        self.last_action_time = 0
        self.action_interval = 2.0  # Minimum time between actions (seconds)
        
        # Retry tracking for each step
        self.step_retry_count = {}  # Dictionary to track retry count per step
        self.max_retries = MAX_RETRY_COUNT
        
        # Flag to track if we've already tapped enter button (skip tap screen/enter checks after)
        self.entered_after_tap = False
        
        # Screen dimensions (will be detected)
        self.screen_width = 0
        self.screen_height = 0
        
        # Gameplay parameters
        self.auto_attack_enabled = True
        self.auto_collect_enabled = True
        self.auto_potion_enabled = True
        self.auto_quest_enabled = True
        
        # Action sequences
        self.action_queue = []
        
        # Last action tracking
        self.last_action = None
        
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
        self.action_queue.clear()
        
    def _detect_screen_size(self):
        """Detect device screen dimensions"""
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
                # Game is not running, try to launch it
                step_name = "game_launch"
                logger.info("Game is not running, attempting to launch...")
                
                if self._launch_game_if_not_running():
                    # Wait for game to start
                    logger.info(f"Game launch attempted, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s for game to start...")
                    time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                    # Check again
                    is_running, package = self.game_detector.is_lineage2m_running()
                    if not is_running:
                        if not self._check_and_reset_retry(step_name):
                            return  # Bot stopped due to max retries
                        logger.warning(f"Game launch attempted but still not running, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s...")
                        time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                        return
                    else:
                        # Success - reset retry counter
                        self._reset_retry(step_name)
                else:
                    if not self._check_and_reset_retry(step_name):
                        return  # Bot stopped due to max retries
                    logger.warning(f"Failed to launch game, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s...")
                    time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                    return
            
            # Skip tap screen and enter button checks if we've already entered after tapping
            if not self.entered_after_tap:
                # Check for "Tap screen" text/image and tap if found (non-blocking check)
                step_name = "tap_screen"
                if self._check_and_tap_tap_screen():
                    # Success - reset retry counter
                    self._reset_retry(step_name)
                    
                    # After tapping "Tap screen", wait for screen transition
                    logger.info(f"Tapped 'Tap screen', waiting {DEFAULT_SCREEN_TRANSITION_TIME}s for screen transition...")
                    time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                    
                    # Wait for enter button to appear and tap it (blocking wait)
                    step_name = "enter_button"
                    logger.info("Waiting for enter_button.png to appear...")
                    if self._wait_and_tap_enter_button(max_wait_time=15.0, check_interval=0.5):
                        # Success - reset retry counter
                        self._reset_retry(step_name)
                        
                        # After tapping enter button, wait 30 seconds without checking for anything
                        logger.info(f"Enter button tapped, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s before detecting game state...")
                        time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
                        
                        # Mark that we've entered after tapping - skip tap screen/enter checks from now on
                        self.entered_after_tap = True
                        
                        # Now try to detect current game state (skip tap screen and enter button checks)
                        logger.info("Detecting current game state after enter button tap...")
                        # Continue to game state detection below (don't return here)
                    else:
                        if not self._check_and_reset_retry(step_name):
                            return  # Bot stopped due to max retries
                        logger.warning("Enter button not found within timeout, continuing...")
                        return
                else:
                    # Tap screen not detected - check retry count
                    if not self._check_and_reset_retry(step_name):
                        return  # Bot stopped due to max retries
            else:
                # Already entered after tapping, skip tap screen and enter button checks
                logger.debug("Already tapped enter button, skipping tap screen/enter button checks, proceeding to game state detection...")
            
            # Detect current game state
            game_state = self.game_detector.detect_game_state()
            self.current_state = game_state.get('status', 'unknown')
            
            # Release screenshot from game_state if it exists (to free memory)
            if 'screenshot' in game_state:
                del game_state['screenshot']
            
            # Execute appropriate actions based on game state
            if self.current_state == 'in_game':
                # Success - reset any retry counters and reset entered flag
                self._reset_retry("game_launch")
                self._reset_retry("tap_screen")
                self._reset_retry("enter_button")
                self.entered_after_tap = False  # Reset flag for next session
                self._execute_in_game_actions()
            elif self.current_state == 'main_menu':
                self._handle_main_menu()
            elif self.current_state == 'loading_screen':
                logger.info(f"Waiting for loading screen to complete... ({DEFAULT_SCREEN_TRANSITION_TIME}s)")
                time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
            elif self.current_state == 'select_server':
                # Already handled by _wait_and_tap_tap_screen
                logger.debug(f"Waiting for server selection... ({DEFAULT_SCREEN_TRANSITION_TIME}s)")
                time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
            elif self.current_state == 'in_game_with_UI':
                # Success - reset any retry counters and reset entered flag
                self._reset_retry("game_launch")
                self._reset_retry("tap_screen")
                self._reset_retry("enter_button")
                self.entered_after_tap = False  # Reset flag for next session
                self._execute_in_game_actions()
            else:
                logger.debug(f"Unknown game state: {self.current_state}, waiting {DEFAULT_SCREEN_TRANSITION_TIME}s...")
                time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
            
            # Periodic garbage collection to free memory (every 10 loops)
            if not hasattr(self, '_loop_count'):
                self._loop_count = 0
            self._loop_count += 1
            if self._loop_count % 10 == 0:
                gc.collect()
                
        except Exception as e:
            step_name = "game_loop_error"
            logger.error(f"Error in game loop: {e}", exc_info=True)
            if not self._check_and_reset_retry(step_name):
                return  # Bot stopped due to max retries
            time.sleep(DEFAULT_SCREEN_TRANSITION_TIME)
    
    def _execute_in_game_actions(self):
        """Execute actions when in game"""
        current_time = time.time()
        
        # Check if enough time has passed since last action
        if current_time - self.last_action_time < self.action_interval:
            return
        
        # Priority 1: Check health and use potions if needed
        if self.auto_potion_enabled:
            health_status = self._check_and_use_potions()
            if STATE_MONITOR_AVAILABLE and self.device_id and health_status:
                device_state_monitor.update_game_state(
                    self.device_id,
                    health_status=health_status
                )
        
        # Priority 2: Auto-attack if enabled
        if self.auto_attack_enabled:
            self._auto_attack()
        
        # Priority 3: Collect items if enabled
        if self.auto_collect_enabled:
            self._collect_items()
        
        # Priority 4: Handle quests/auto-quest if enabled
        if self.auto_quest_enabled:
            self._handle_quests()
        
        self.last_action_time = current_time
        
    def _launch_game_if_not_running(self) -> bool:
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
            
            # Check if any package is installed
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
    
    def _check_and_tap_tap_screen(self) -> bool:
        """
        Check for "Tap screen" text or image and tap if found (non-blocking single check)
        
        Returns:
            True if "Tap screen" was detected and tapped, False otherwise
        """
        try:
            # Take screenshot
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                return False
            
            # Check for "Tap screen" text using OCR
            tap_screen_detected = self.game_detector._detect_tap_screen_text(screenshot)
            
            if tap_screen_detected:
                logger.info("'Tap screen' text detected, attempting to find position and tap...")
                
                # Try to find tap screen position using OCR results
                tap_position = self._find_tap_screen_position(screenshot)
                
                if tap_position:
                    x, y = tap_position
                    logger.info(f"Tapping at detected position: ({x}, {y})")
                    self.adb.tap(x, y)
                    
                    if STATE_MONITOR_AVAILABLE and self.device_id:
                        device_state_monitor.update_game_state(
                            self.device_id,
                            action="tap_screen"
                        )
                    
                    return True
                else:
                    # If text detected but position not found, tap center of screen
                    logger.info("'Tap screen' text detected but position not found, tapping center of screen")
                    center_x = self.screen_width // 2 if self.screen_width > 0 else 540
                    center_y = self.screen_height // 2 if self.screen_height > 0 else 960
                    self.adb.tap(center_x, center_y)
                    
                    if STATE_MONITOR_AVAILABLE and self.device_id:
                        device_state_monitor.update_game_state(
                            self.device_id,
                            action="tap_screen_center"
                        )
                    
                    return True
            
            # Also check for "Tap screen" image template if available
            try:
                from ..utils.template_matcher import TemplateMatcher
                template_matcher = TemplateMatcher()
                
                # Try to find "tap_screen" template
                tap_result = template_matcher.find_template(screenshot, "tap_screen.png", multi_scale=True)
                if tap_result:
                    x, y, confidence = tap_result
                    logger.info(f"'Tap screen' image detected at ({x}, {y}) with confidence {confidence:.3f}, tapping...")
                    self.adb.tap(x, y)
                    
                    if STATE_MONITOR_AVAILABLE and self.device_id:
                        device_state_monitor.update_game_state(
                            self.device_id,
                            action="tap_screen_image"
                        )
                    
                    return True
            except Exception as e:
                logger.debug(f"Template matching for tap screen failed: {e}")
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking tap screen: {e}")
            return False
    
    def _wait_and_tap_enter_button(self, max_wait_time: float = 15.0, check_interval: float = 0.5) -> bool:
        """
        Wait for enter_button.png template to appear, then tap it
        
        Args:
            max_wait_time: Maximum time to wait in seconds (default: 15)
            check_interval: Interval between checks in seconds (default: 0.5)
            
        Returns:
            True if enter button was detected and tapped, False otherwise
        """
        try:
            from ..utils.template_matcher import TemplateMatcher
            template_matcher = TemplateMatcher()
            
            start_time = time.time()
            check_count = 0
            
            logger.info(f"Waiting for enter_button.png to appear (max {max_wait_time}s, checking every {check_interval}s)...")
            
            while time.time() - start_time < max_wait_time:
                check_count += 1
                elapsed = time.time() - start_time
                
                # Take screenshot
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.debug(f"Could not take screenshot for enter button detection (check #{check_count}, {elapsed:.1f}s elapsed)")
                    time.sleep(check_interval)
                    continue
                
                # Try to find enter_button.png template
                try:
                    enter_result = template_matcher.find_template(screenshot, "enter_button.png", multi_scale=True)
                    if enter_result:
                        x, y, confidence = enter_result
                        logger.info(f"[OK] enter_button.png detected at ({x}, {y}) with confidence {confidence:.3f} (after {elapsed:.1f}s), tapping...")
                        self.adb.tap(x, y)
                        
                        if STATE_MONITOR_AVAILABLE and self.device_id:
                            device_state_monitor.update_game_state(
                                self.device_id,
                                action="tap_enter_button"
                            )
                        
                        return True
                    else:
                        # Log progress every 2 seconds
                        if check_count % 4 == 0:  # Every 2 seconds (4 checks * 0.5s)
                            logger.debug(f"Still waiting for enter_button.png... ({elapsed:.1f}s elapsed, check #{check_count})")
                except Exception as e:
                    logger.debug(f"Template matching for enter button failed (check #{check_count}): {e}")
                
                # Wait before next check
                time.sleep(check_interval)
            
            elapsed = time.time() - start_time
            logger.warning(f"enter_button.png not detected within {max_wait_time} seconds (checked {check_count} times, {elapsed:.1f}s elapsed)")
            return False
            
        except ImportError as e:
            logger.warning(f"TemplateMatcher not available for enter button detection: {e}")
            return False
        except Exception as e:
            logger.error(f"Error waiting for enter button: {e}", exc_info=True)
            return False
    
    def _find_tap_screen_position(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the position of "Tap screen" text using OCR
        
        Args:
            screenshot: Screenshot image
            
        Returns:
            Tuple of (x, y) coordinates if found, None otherwise
        """
        try:
            # Try to import EasyOCR
            try:
                import easyocr
            except ImportError:
                logger.debug("EasyOCR not available for position detection")
                return None
            
            # Use shared OCR reader to avoid multiple instances
            from ..utils.ocr_reader import shared_ocr_reader
            ocr_reader = shared_ocr_reader.get_reader()
            if ocr_reader is None:
                return None
            
            # Downsample image for OCR to reduce memory usage
            height, width = screenshot.shape[:2]
            max_dimension = 1920
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                screenshot_resized = cv2.resize(screenshot, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                screenshot_resized = screenshot
            
            # Convert BGR to RGB for EasyOCR
            rgb_image = cv2.cvtColor(screenshot_resized, cv2.COLOR_BGR2RGB)
            
            # Perform OCR on the resized screenshot
            results = ocr_reader.readtext(rgb_image, detail=1, paragraph=False)
            
            # Search for "Tap screen" variations
            tap_screen_variations = [
                'tap screen',
                'tap to screen',
                'tap the screen',
                'tap screen to',
                'tap',
                'tap to start',
                'tap to begin'
            ]
            
            for (bbox, text, confidence) in results:
                text_lower = text.lower().strip()
                for variation in tap_screen_variations:
                    if variation in text_lower:
                        # Calculate center of bounding box
                        # bbox is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        if len(bbox) >= 4:
                            # Get all x and y coordinates
                            xs = [point[0] for point in bbox]
                            ys = [point[1] for point in bbox]
                            
                            # Calculate center
                            center_x = int(sum(xs) / len(xs))
                            center_y = int(sum(ys) / len(ys))
                            
                            logger.info(f"Found 'Tap screen' text '{text}' at ({center_x}, {center_y})")
                            return (center_x, center_y)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error finding tap screen position: {e}")
            return None
    
    def _check_and_use_potions(self) -> Optional[str]:
        """Check HP/MP and use potions if needed
        
        Returns:
            Health status string (e.g., 'low_hp', 'low_mp', 'healthy')
        """
        try:
            # Take screenshot for analysis
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                return None
            
            health_status = "healthy"
            
            # Analyze health bar area (typically at top of screen)
            # Health bar is usually in top-left or top-center area
            health_region = screenshot[0:self.screen_height//10, 0:self.screen_width//3]
            
            # Simple heuristic: Check if health bar area is mostly red (low health)
            # Convert to HSV for better color detection
            hsv_region = cv2.cvtColor(health_region, cv2.COLOR_BGR2HSV)
            
            # Red color range in HSV (for low health indicator)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            
            # Create mask for red pixels
            red_mask = cv2.inRange(hsv_region, lower_red, upper_red)
            red_ratio = np.sum(red_mask > 0) / (health_region.shape[0] * health_region.shape[1])
            
            # If significant red detected (low health), use HP potion
            if red_ratio > 0.3:
                logger.info("Low health detected, using HP potion...")
                self._use_hp_potion()
                health_status = "low_hp"
                time.sleep(0.5)
            
            # Check MP (mana) - typically blue or green bar
            # For now, we'll use a simpler approach - check overall brightness
            mp_region = screenshot[0:self.screen_height//10, self.screen_width//3:self.screen_width*2//3]
            mp_brightness = np.mean(cv2.cvtColor(mp_region, cv2.COLOR_BGR2GRAY))
            
            # If MP bar is dark (low mana), use MP potion
            if mp_brightness < 100:
                logger.info("Low MP detected, using MP potion...")
                self._use_mp_potion()
                if health_status == "healthy":
                    health_status = "low_mp"
                else:
                    health_status = "low_hp_mp"
                time.sleep(0.5)
            
            return health_status
                
        except Exception as e:
            logger.debug(f"Error checking potions: {e}")
            return None
    
    def _use_hp_potion(self):
        """Use HP potion (typically F1 key or button)"""
        # HP potion is usually mapped to a specific key or button
        # Common locations: Top of screen, quick slot, or number key
        # For now, we'll tap a common HP potion location (adjust based on your UI)
        # This is a placeholder - you'll need to adjust coordinates based on your game UI
        
        # Option 1: Tap quick slot button (if HP potion is in quick slot)
        # This is typically in bottom-right area
        potion_x = int(self.screen_width * 0.85)
        potion_y = int(self.screen_height * 0.85)
        
        self._tap(potion_x, potion_y)
        self.last_action = "use_hp_potion"
        
        # Update state monitor
        if STATE_MONITOR_AVAILABLE and self.device_id:
            device_state_monitor.update_game_state(
                self.device_id,
                action="use_hp_potion",
                health_status="used_hp_potion"
            )
        
        logger.info(f"Used HP potion at ({potion_x}, {potion_y})")
    
    def _use_mp_potion(self):
        """Use MP potion (typically F2 key or button)"""
        # Similar to HP potion, adjust coordinates
        potion_x = int(self.screen_width * 0.90)
        potion_y = int(self.screen_height * 0.85)
        
        self._tap(potion_x, potion_y)
        self.last_action = "use_mp_potion"
        
        # Update state monitor
        if STATE_MONITOR_AVAILABLE and self.device_id:
            device_state_monitor.update_game_state(
                self.device_id,
                action="use_mp_potion",
                health_status="used_mp_potion"
            )
        
        logger.info(f"Used MP potion at ({potion_x}, {potion_y})")
    
    def _auto_attack(self):
        """Auto-attack nearby enemies"""
        try:
            # Take screenshot to detect enemies
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                return
            
            # Attack button is typically in bottom-center or bottom-right
            # Common location for auto-attack button
            attack_x = int(self.screen_width * 0.5)
            attack_y = int(self.screen_height * 0.9)
            
            # Tap attack button
            self._tap(attack_x, attack_y)
            self.last_action = "auto_attack"
            
            # Update state monitor
            if STATE_MONITOR_AVAILABLE and self.device_id:
                device_state_monitor.update_game_state(
                    self.device_id,
                    action="auto_attack"
                )
            
            logger.debug(f"Auto-attack executed at ({attack_x}, {attack_y})")
            
            # Small delay after attack
            time.sleep(0.3)
            
        except Exception as e:
            logger.debug(f"Error in auto-attack: {e}")
    
    def _collect_items(self):
        """Collect items on the ground"""
        try:
            # Take screenshot to detect items
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                return
            
            # Look for item indicators on screen
            # Items are typically marked with icons or text
            # For now, we'll use a simple approach: tap center area where items might be
            
            # Item collection is usually done by tapping the item or using collect button
            # Common locations: center of screen or bottom area
            collect_x = int(self.screen_width * 0.5 + random.randint(-50, 50))
            collect_y = int(self.screen_height * 0.6 + random.randint(-50, 50))
            
            self._tap(collect_x, collect_y)
            self.last_action = "collect_items"
            
            # Update state monitor
            if STATE_MONITOR_AVAILABLE and self.device_id:
                device_state_monitor.update_game_state(
                    self.device_id,
                    action="collect_items"
                )
            
            logger.debug(f"Attempted to collect item at ({collect_x}, {collect_y})")
            
            time.sleep(0.2)
            
        except Exception as e:
            logger.debug(f"Error collecting items: {e}")
    
    def _handle_quests(self):
        """Handle quest-related actions"""
        try:
            # Check for quest completion notifications
            # Quest notifications are typically at top or side of screen
            
            # Take screenshot
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                return
            
            # Look for quest notification indicators
            # This is a placeholder - implement based on your game's UI
            
            # For now, we'll just check periodically
            # Quest completion is often indicated by specific UI elements
            
        except Exception as e:
            logger.debug(f"Error handling quests: {e}")
    
    def _handle_main_menu(self):
        """Handle actions when in main menu"""
        try:
            logger.info("In main menu, attempting to enter game...")
            
            # Tap "Start" or "Enter Game" button
            # Common locations: center of screen
            start_x = int(self.screen_width * 0.5)
            start_y = int(self.screen_height * 0.7)
            
            self._tap(start_x, start_y)
            time.sleep(2)
            
        except Exception as e:
            logger.debug(f"Error handling main menu: {e}")
    
    def _tap(self, x: int, y: int, duration: float = 0.1):
        """
        Execute a tap action at specified coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Tap duration in seconds
        """
        try:
            success, output = self.adb.execute_adb_command([
                'shell', 'input', 'tap', str(x), str(y)
            ])
            
            if success:
                logger.debug(f"Tapped at ({x}, {y})")
                time.sleep(duration)
            else:
                logger.warning(f"Failed to tap at ({x}, {y}): {output}")
                
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
    
    def set_auto_attack(self, enabled: bool):
        """Enable/disable auto-attack"""
        self.auto_attack_enabled = enabled
        logger.info(f"Auto-attack: {'enabled' if enabled else 'disabled'}")
    
    def set_auto_collect(self, enabled: bool):
        """Enable/disable auto-collect"""
        self.auto_collect_enabled = enabled
        logger.info(f"Auto-collect: {'enabled' if enabled else 'disabled'}")
    
    def set_auto_potion(self, enabled: bool):
        """Enable/disable auto-potion"""
        self.auto_potion_enabled = enabled
        logger.info(f"Auto-potion: {'enabled' if enabled else 'disabled'}")
    
    def set_auto_quest(self, enabled: bool):
        """Enable/disable auto-quest"""
        self.auto_quest_enabled = enabled
        logger.info(f"Auto-quest: {'enabled' if enabled else 'disabled'}")

