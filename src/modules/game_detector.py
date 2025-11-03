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
            foreground_app = self.adb.get_foreground_app()
            
            # Check if foreground app is Lineage 2M
            if foreground_app:
                for package in self.lineage2m_packages:
                    if package in foreground_app:
                        return True, package
            
            # Check if any Lineage 2M package is running (even if not foreground)
            for package in self.lineage2m_packages:
                if self.adb.is_app_running(package):
                    return True, package
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking if Lineage 2M is running: {e}")
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
            
            # Basic game state detection
            state = {
                'status': 'unknown',
                'screenshot_taken': True,
                'screen_size': screenshot.shape[:2],
                'timestamp': self._get_current_timestamp(),
                'message': 'Screenshot captured successfully'
            }
            
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
            target_package = package_name or self.lineage2m_packages[0]
            
            logger.info(f"Launching Lineage 2M: {target_package}")
            
            success, output = self.adb.execute_adb_command([
                'shell', 'monkey', '-p', target_package, '-c', 
                'android.intent.category.LAUNCHER', '1'
            ])
            
            if success:
                logger.info(f"Launch command executed for {target_package}")
                return True
            else:
                logger.error(f"Failed to launch {target_package}: {output}")
                return False
                
        except Exception as e:
            logger.error(f"Error launching game: {e}")
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
            List of installed package names
        """
        try:
            installed_packages = []
            
            for package in self.lineage2m_packages:
                success, output = self.adb.execute_adb_command([
                    'shell', 'pm', 'list', 'packages', package
                ])
                
                if success and package in output:
                    installed_packages.append(package)
            
            logger.info(f"Found {len(installed_packages)} installed Lineage 2M packages")
            return installed_packages
            
        except Exception as e:
            logger.error(f"Error getting installed packages: {e}")
            return []