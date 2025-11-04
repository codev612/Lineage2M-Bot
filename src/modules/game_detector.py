"""
Game Detector Module - Detects Lineage 2M game state and status
Handles game detection, state analysis, and game-specific operations
"""

from typing import Tuple, Optional, Dict, List
import re
import cv2
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import GameConfig
from ..utils.exceptions import GameStateError

logger = get_logger(__name__)

class GameDetector:
    """
    Detects Lineage 2M game state and provides game-specific functionality
    """
    
    def __init__(self, adb_manager, game_config: GameConfig):
        """
        Initialize game detector
        
        Args:
            adb_manager: ADB manager instance
            game_config: Game configuration
        """
        self.adb = adb_manager
        self.config = game_config
        
        # Default Lineage 2M package names if not configured
        self.lineage2m_packages = self.config.packages or [
            'com.ncsoft.lineage2m',
            'com.ncsoft.lineage2m.android',
            'com.ncsoft.lineage2m.global',
            'com.ncsoft.lineage2m.sea',
            'com.ncsoft.lineage2m.kr'
        ]
    
    def is_lineage2m_running(self) -> Tuple[bool, Optional[str]]:
        """
        Check if Lineage 2M is running and return package name if found
        
        Returns:
            Tuple of (is_running, package_name)
        """
        try:
            # First, check foreground app
            foreground_app = self.adb.get_foreground_app()
            logger.debug(f"Foreground app: {foreground_app}")
            
            # Check if foreground app is Lineage 2M (exact or partial match)
            if foreground_app:
                for package in self.lineage2m_packages:
                    # Check for exact match or if package is contained in foreground app
                    if package == foreground_app or package in foreground_app:
                        logger.info(f"Lineage 2M detected as foreground app: {package} (foreground: {foreground_app})")
                        return True, package
                    
                # Also check reverse - if foreground app contains any lineage2m package
                if 'lineage2m' in foreground_app.lower():
                    # Find the matching package
                    for package in self.lineage2m_packages:
                        if package in foreground_app:
                            logger.info(f"Lineage 2M detected in foreground app: {package} (foreground: {foreground_app})")
                            return True, package
                    # If no exact match but contains lineage2m, return the foreground app itself
                    logger.info(f"Lineage 2M detected (variant): {foreground_app}")
                    return True, foreground_app
            
            # Check if any Lineage 2M package is running (even if not foreground)
            logger.debug(f"Checking running packages: {self.lineage2m_packages}")
            for package in self.lineage2m_packages:
                is_running = self.adb.is_app_running(package)
                logger.debug(f"Package {package} running check: {is_running}")
                if is_running:
                    logger.info(f"Lineage 2M detected as running (background): {package}")
                    return True, package
            
            # Additional check: List all running processes and search for lineage2m
            try:
                success, output = self.adb.execute_adb_command(['shell', 'ps', '-A'])
                if success and output:
                    # Search for lineage2m in process list
                    for line in output.split('\n'):
                        if 'lineage2m' in line.lower():
                            # Try to extract package name from process line
                            import re
                            # Look for package pattern in the line
                            match = re.search(r'([a-zA-Z0-9._]*lineage2m[a-zA-Z0-9._]*)', line, re.IGNORECASE)
                            if match:
                                found_package = match.group(1)
                                logger.info(f"Lineage 2M detected in process list: {found_package}")
                                return True, found_package
            except Exception as e:
                logger.debug(f"Error checking process list: {e}")
            
            logger.debug("Lineage 2M not detected as running")
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking if Lineage 2M is running: {e}", exc_info=True)
            return False, None
    
    def detect_game_state(self) -> Dict:
        """
        Detect current game state using screenshot analysis
        
        Returns:
            Dictionary containing game state information
        """
        try:
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                return {
                    'status': 'error', 
                    'message': 'Could not take screenshot',
                    'screenshot_taken': False
                }
            
            # Check for "Tap screen" text first (common game startup state)
            tap_screen_detected = self._detect_tap_screen_text(screenshot)
            
            # Basic game state detection
            state = {
                'status': 'select_server' if tap_screen_detected else 'unknown',
                'screenshot_taken': True,
                'screen_size': screenshot.shape[:2],
                'timestamp': self._get_current_timestamp(),
                'message': 'Screenshot captured successfully',
                'tap_screen_detected': tap_screen_detected
            }
            
            # If "Tap screen" is detected, we're in server selection state
            if tap_screen_detected:
                logger.info("'Tap screen' text detected - game is in server selection state")
                return state
            
            # Analyze screenshot for game elements
            game_elements = self._analyze_screenshot(screenshot)
            state.update(game_elements)
            
            return state
            
        except Exception as e:
            logger.error(f"Error detecting game state: {e}")
            return {
                'status': 'error',
                'message': f'Error detecting game state: {e}',
                'screenshot_taken': False
            }
    
    def _detect_tap_screen_text(self, screenshot: np.ndarray) -> bool:
        """
        Detect "Tap screen" text on screenshot using OCR
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            True if "Tap screen" text is detected, False otherwise
        """
        try:
            # Try to import EasyOCR
            try:
                import easyocr
            except ImportError:
                logger.debug("EasyOCR not available for 'Tap screen' detection")
                return False
            
            # Initialize OCR reader if not already done
            if not hasattr(self, '_ocr_reader') or self._ocr_reader is None:
                try:
                    logger.debug("Initializing EasyOCR reader for 'Tap screen' detection...")
                    self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                    logger.debug("EasyOCR reader initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize EasyOCR: {e}")
                    return False
            
            # Convert BGR to RGB for EasyOCR
            rgb_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            
            # Perform OCR on the entire screenshot
            # Use paragraph=False to get individual text detections
            results = self._ocr_reader.readtext(rgb_image, detail=1, paragraph=False)
            
            # Search for "Tap screen" or variations
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
                # Check if any variation matches
                for variation in tap_screen_variations:
                    if variation in text_lower:
                        logger.info(f"Detected 'Tap screen' text: '{text}' (confidence: {confidence:.3f})")
                        return True
            
            logger.debug("'Tap screen' text not detected in screenshot")
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting 'Tap screen' text: {e}")
            return False
    
    def _analyze_screenshot(self, screenshot: np.ndarray) -> Dict:
        """
        Analyze screenshot for game-specific elements
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            Dictionary with detected elements
        """
        analysis = {
            'ui_elements': [],
            'colors': {},
            'text_detected': False,
            'menu_state': 'unknown'
        }
        
        try:
            # Convert to different color spaces for analysis
            hsv_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            gray_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Analyze dominant colors
            analysis['colors'] = self._analyze_colors(screenshot)
            
            # Detect UI elements (basic edge detection)
            edges = cv2.Canny(gray_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count significant contours (potential UI elements)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
            analysis['ui_elements'] = len(significant_contours)
            
            # Basic game state detection based on colors and layout
            analysis['menu_state'] = self._detect_menu_state(screenshot, hsv_image)
            
        except Exception as e:
            logger.warning(f"Error analyzing screenshot: {e}")
        
        return analysis
    
    def _analyze_colors(self, image: np.ndarray) -> Dict:
        """
        Analyze dominant colors in the screenshot
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary with color analysis
        """
        try:
            # Resize image for faster processing
            small_image = cv2.resize(image, (100, 100))
            
            # Calculate mean colors
            mean_color = np.mean(small_image, axis=(0, 1))
            
            # Convert to different color spaces
            hsv_mean = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            return {
                'mean_bgr': mean_color.tolist(),
                'mean_hsv': hsv_mean.tolist(),
                'brightness': float(np.mean(cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY))),
                'saturation': float(hsv_mean[1])
            }
        except Exception as e:
            logger.warning(f"Error analyzing colors: {e}")
            return {}
    
    def _detect_menu_state(self, image: np.ndarray, hsv_image: np.ndarray) -> str:
        """
        Detect basic menu state based on image analysis
        
        Args:
            image: Original BGR image
            hsv_image: HSV converted image
            
        Returns:
            Detected menu state as string
        """
        try:
            height, width = image.shape[:2]
            
            # Sample different regions of the screen
            top_region = image[0:height//4, :]
            bottom_region = image[3*height//4:height, :]
            center_region = image[height//4:3*height//4, width//4:3*width//4]
            
            # Analyze regions
            top_brightness = np.mean(cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY))
            bottom_brightness = np.mean(cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY))
            center_brightness = np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY))
            
            # Basic heuristics for menu detection
            if top_brightness > 200 and bottom_brightness > 200:
                return 'main_menu'
            elif bottom_brightness > top_brightness + 50:
                return 'in_game_with_ui'
            elif center_brightness < 50:
                return 'loading_screen'
            else:
                return 'in_game'
                
        except Exception as e:
            logger.warning(f"Error detecting menu state: {e}")
            return 'unknown'
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def launch_game(self, package_name: str = None) -> bool:
        """
        Launch Lineage 2M game
        
        Args:
            package_name: Specific package to launch, uses first available if None
            
        Returns:
            True if launch command executed successfully
        """
        try:
            # If no package specified, try to find installed packages first
            if not package_name:
                installed = self.get_installed_lineage2m_packages()
                if installed:
                    target_package = installed[0]
                    logger.info(f"No package specified, using first installed: {target_package}")
                else:
                    target_package = self.lineage2m_packages[0]
                    logger.warning(f"No installed packages found, trying default: {target_package}")
            else:
                target_package = package_name
            
            logger.info(f"Attempting to launch Lineage 2M: {target_package}")
            
            # Verify package is installed first (must use exact match)
            # pm list packages <name> does partial matching, so we need to check all packages
            success, output = self.adb.execute_adb_command([
                'shell', 'pm', 'list', 'packages'
            ])
            
            if not success:
                logger.error(f"Failed to list packages")
                return False
            
            # Check for exact package match (package:com.ncsoft.lineage2mnu)
            exact_match = f"package:{target_package}" in output
            if not exact_match:
                # Try to find the actual installed package (might be a variant)
                import re
                pattern = re.compile(rf'package:({re.escape(target_package)}[^\s]*)')
                matches = pattern.findall(output)
                if matches:
                    actual_package = matches[0]
                    logger.warning(f"Package {target_package} not found, but found similar package: {actual_package}")
                    logger.info(f"Using actual installed package: {actual_package}")
                    target_package = actual_package
                else:
                    logger.error(f"Package {target_package} is not installed (exact match)")
                    return False
            
            logger.info(f"Package {target_package} is installed, proceeding with launch")
            
            # Method 1: Try simple monkey command first (most reliable for most apps)
            try:
                logger.info(f"Trying simple monkey command: {target_package}")
                success, output = self.adb.execute_adb_command([
                    'shell', 'monkey', '-p', target_package, '1'
                ])
                
                if success:
                    logger.info(f"Game launched successfully using simple monkey: {target_package}")
                    logger.info(f"Command output: {output}")
                    return True
                else:
                    logger.warning(f"Simple monkey command failed: {output}")
            except Exception as e:
                logger.warning(f"Simple monkey command exception: {e}")
            
            # Method 2: Try to get launcher activity from dumpsys window (current focused app)
            try:
                # Get current window to find launcher activity pattern
                success, output = self.adb.execute_adb_command([
                    'shell', 'dumpsys', 'window', 'windows'
                ])
                
                if success and output:
                    import re
                    # Look for activity pattern like: package/activity
                    activity_pattern = re.search(rf'{re.escape(target_package)}/([a-zA-Z0-9._]+Activity)', output)
                    if activity_pattern:
                        activity = f"{target_package}/{activity_pattern.group(1)}"
                        logger.info(f"Found activity from window dump: {activity}")
                        
                        # Try launching with this activity
                        success, output = self.adb.execute_adb_command([
                            'shell', 'am', 'start', '-n', activity
                        ])
                        
                        if success:
                            logger.info(f"Game launched successfully using am start with activity: {activity}")
                            return True
            except Exception as e:
                logger.debug(f"Could not get activity from window dump: {e}")
            
            # Method 3: Try am start with main activity (get from dumpsys package)
            try:
                # Get launcher activity using dumpsys package
                success, output = self.adb.execute_adb_command([
                    'shell', 'dumpsys', 'package', target_package
                ])
                
                if success and output:
                    # Try to extract launcher activity from package dump
                    import re
                    # Look for activity with MAIN/LAUNCHER intent filter
                    # Pattern: Activity name with MAIN action and LAUNCHER category
                    patterns = [
                        r'([a-zA-Z0-9._]+\.[a-zA-Z0-9._]+Activity)\s+\w+ filter.*?android\.intent\.action\.MAIN',
                        r'Activity.*?([a-zA-Z0-9._]+\.[a-zA-Z0-9._]+Activity).*?MAIN.*?LAUNCHER',
                        r'([a-zA-Z0-9._]+/[a-zA-Z0-9._]+Activity).*?MAIN.*?LAUNCHER'
                    ]
                    
                    activity = None
                    for pattern in patterns:
                        match = re.search(pattern, output, re.DOTALL)
                        if match:
                            activity = match.group(1)
                            # If activity doesn't have package prefix, add it
                            if not activity.startswith(target_package):
                                if '/' not in activity:
                                    activity = f"{target_package}/{activity}"
                            break
                    
                    if activity:
                        logger.info(f"Found launcher activity: {activity}")
                        
                        # Launch using am start
                        success, output = self.adb.execute_adb_command([
                            'shell', 'am', 'start', '-n', activity
                        ])
                        
                        if success:
                            logger.info(f"Game launched successfully using am start: {target_package}")
                            return True
                        else:
                            logger.debug(f"am start with activity {activity} failed: {output}")
            except Exception as e:
                logger.debug(f"Could not get main activity: {e}")
            
            # Method 4: Try common activity names (especially for Unity/Unreal games like Lineage 2M)
            common_activities = [
                'com.epicgames.ue4.GameActivity',  # Unreal Engine games (Lineage 2M uses this!)
                'com.unity3d.player.UnityPlayerActivity',  # Unity games
                'MainActivity', 
                'SplashActivity', 
                'UnityPlayerActivity', 
                'GameActivity'
            ]
            for activity_name in common_activities:
                try:
                    # Handle both full activity paths and simple names
                    if activity_name.startswith('com.'):
                        # Full activity path - use as package/activity
                        activity = f'{target_package}/{activity_name}'
                    else:
                        # Simple name - prepend package
                        activity = f'{target_package}/{activity_name}'
                    
                    logger.info(f"Trying activity: {activity}")
                    success, output = self.adb.execute_adb_command([
                        'shell', 'am', 'start', '-n', activity
                    ])
                    
                    if success:
                        logger.info(f"Game launched successfully using activity {activity}: {target_package}")
                        logger.info(f"Command output: {output}")
                        # Even if warning says "already running", command succeeded
                        return True
                    else:
                        logger.debug(f"am start with {activity_name} failed: {output}")
                except Exception as e:
                    logger.debug(f"am start with {activity_name} exception: {e}")
                    continue
            
            # Method 5: Try am start with package intent (works if package has default launcher)
            try:
                logger.info(f"Trying am start with package intent: {target_package}")
                success, output = self.adb.execute_adb_command([
                    'shell', 'am', 'start', '-a', 'android.intent.action.MAIN',
                    '-c', 'android.intent.category.LAUNCHER',
                    target_package
                ])
                
                if success and 'Error' not in output:
                    logger.info(f"Game launched successfully using am start with package: {target_package}")
                    logger.info(f"Command output: {output}")
                    return True
                else:
                    logger.warning(f"am start with package failed: {output}")
            except Exception as e:
                logger.warning(f"am start with package exception: {e}")
            
            # Method 6: Try monkey with category (last resort)
            try:
                logger.info(f"Trying monkey with category: {target_package}")
                success, output = self.adb.execute_adb_command([
                    'shell', 'monkey', '-p', target_package, 
                    '-c', 'android.intent.category.LAUNCHER', '1'
                ])
                
                if success:
                    logger.info(f"Game launched successfully using monkey with category: {target_package}")
                    logger.info(f"Command output: {output}")
                    return True
                else:
                    logger.warning(f"Monkey with category failed: {output}")
            except Exception as e:
                logger.warning(f"Monkey with category exception: {e}")
            
            # All methods failed - log detailed error
            logger.error(f"All launch methods failed for {target_package}")
            logger.error(f"Please check:")
            logger.error(f"  1. Package {target_package} is installed: adb shell pm list packages {target_package}")
            logger.error(f"  2. Try manually: adb shell am start -n {target_package}/.MainActivity")
            logger.error(f"  3. Or try: adb shell monkey -p {target_package} 1")
            return False
                
        except Exception as e:
            logger.error(f"Error launching game: {e}", exc_info=True)
            return False
    
    def close_game(self, package_name: str = None) -> bool:
        """
        Close Lineage 2M game
        
        Args:
            package_name: Specific package to close, uses currently running if None
            
        Returns:
            True if close command executed successfully
        """
        try:
            # Find running package if not specified
            if not package_name:
                is_running, running_package = self.is_lineage2m_running()
                if not is_running:
                    logger.info("No Lineage 2M game is currently running")
                    return True
                package_name = running_package
            
            logger.info(f"Closing Lineage 2M: {package_name}")
            
            success, output = self.adb.execute_adb_command([
                'shell', 'am', 'force-stop', package_name
            ])
            
            if success:
                logger.info(f"Close command executed for {package_name}")
                return True
            else:
                logger.error(f"Failed to close {package_name}: {output}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing game: {e}")
            return False
    
    def wait_for_game_start(self, timeout: int = 60) -> bool:
        """
        Wait for Lineage 2M to start
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if game started within timeout
        """
        import time
        
        logger.info(f"Waiting for Lineage 2M to start (timeout: {timeout}s)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            is_running, package = self.is_lineage2m_running()
            if is_running:
                logger.info(f"Lineage 2M started: {package}")
                return True
            
            time.sleep(2)
        
        logger.warning(f"Lineage 2M did not start within {timeout} seconds")
        return False
    
    def get_installed_lineage2m_packages(self) -> List[str]:
        """
        Get list of installed Lineage 2M packages
        
        Returns:
            List of installed package names (exact matches only)
        """
        try:
            installed_packages = []
            
            # Get all packages first to check for exact matches
            success, all_packages_output = self.adb.execute_adb_command([
                'shell', 'pm', 'list', 'packages'
            ])
            
            if not success:
                logger.error("Failed to list all packages")
                return []
            
            # Check known packages for EXACT matches only
            # IMPORTANT: We need to check for exact package:package_name format
            # Split by newlines and check each line to avoid partial matches
            package_lines = all_packages_output.split('\n')
            for package in self.lineage2m_packages:
                # Check for exact match in format "package:com.ncsoft.lineage2m\n" or "package:com.ncsoft.lineage2m"
                exact_match = False
                for line in package_lines:
                    # Match exactly "package:com.ncsoft.lineage2m" (not "package:com.ncsoft.lineage2mnu")
                    if line.strip() == f"package:{package}":
                        exact_match = True
                        break
                
                if exact_match:
                    installed_packages.append(package)
                    logger.debug(f"Found exact match for known package: {package}")
                else:
                    logger.debug(f"No exact match for known package: {package} (might have variant like lineage2mnu)")
            
            # Also search for any package containing "lineage2m" to catch variants like "lineage2mnu"
            # This finds packages that are NOT in the known list
            try:
                import re
                # Look for any package containing lineage2m (case insensitive)
                pattern = re.compile(r'package:([^\s]+lineage2m[^\s]*)', re.IGNORECASE)
                matches = pattern.findall(all_packages_output)
                for match in matches:
                    if match not in installed_packages:
                        installed_packages.append(match)
                        logger.info(f"Found additional Lineage 2M package: {match}")
            except Exception as e:
                logger.debug(f"Error searching for lineage2m packages: {e}")
            
            # Sort: put exact matches from known list first, then variants
            # This ensures we prefer known packages over variants
            known_packages = [p for p in installed_packages if p in self.lineage2m_packages]
            variant_packages = [p for p in installed_packages if p not in self.lineage2m_packages]
            installed_packages = known_packages + variant_packages
            
            logger.info(f"Found {len(installed_packages)} installed Lineage 2M packages: {installed_packages}")
            return installed_packages
            
        except Exception as e:
            logger.error(f"Error getting installed packages: {e}")
            return []