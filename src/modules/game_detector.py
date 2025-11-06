"""
Game Detector Module - Detects Lineage 2M game state and status
Handles game detection, state analysis, and game-specific operations
"""

from typing import Tuple, Optional, Dict, List, Any
import re
import cv2
import numpy as np
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import GameConfig
from ..utils.exceptions import GameStateError
from ..utils.template_matcher import TemplateMatcher

logger = get_logger(__name__)

# Helper functions for color conversions with OpenCV fallbacks
def _bgr_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale - fallback if cv2.cvtColor doesn't exist"""
    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2GRAY'):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Manual BGR to grayscale conversion using standard formula
        # gray = 0.114*B + 0.587*G + 0.299*R
        if len(image.shape) == 3:
            b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = (0.114 * b.astype(np.float32) + 
                    0.587 * g.astype(np.float32) + 
                    0.299 * r.astype(np.float32)).astype(np.uint8)
            return gray
        else:
            return image

def _bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV - fallback if cv2.cvtColor doesn't exist"""
    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2HSV'):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        # Manual BGR to HSV conversion
        # This is a simplified version - for full accuracy, OpenCV's implementation is preferred
        if len(image.shape) != 3:
            raise ValueError("BGR to HSV conversion requires 3-channel image")
        
        bgr = image.astype(np.float32) / 255.0
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        
        # Calculate V (Value/Brightness)
        v = np.maximum(np.maximum(b, g), r)
        
        # Calculate S (Saturation)
        min_val = np.minimum(np.minimum(b, g), r)
        delta = v - min_val
        s = np.where(v == 0, 0, delta / (v + 1e-10))
        
        # Calculate H (Hue)
        h = np.zeros_like(v)
        
        # Red channel is max
        mask_r = (v == r) & (delta != 0)
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-10)) % 6)
        
        # Green channel is max
        mask_g = (v == g) & (delta != 0)
        h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-10)) + 2)
        
        # Blue channel is max
        mask_b = (v == b) & (delta != 0)
        h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-10)) + 4)
        
        # Normalize to OpenCV range: H[0-179], S[0-255], V[0-255]
        h = (h / 2).astype(np.uint8)  # H: 0-180 -> 0-179
        s = (s * 255).astype(np.uint8)  # S: 0-1 -> 0-255
        v = (v * 255).astype(np.uint8)  # V: 0-1 -> 0-255
        
        hsv = np.stack([h, s, v], axis=2)
        return hsv

def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image - fallback if cv2.resize doesn't exist"""
    if hasattr(cv2, 'resize'):
        return cv2.resize(image, size)
    else:
        # Fallback: Use PIL to resize
        from PIL import Image
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            # BGR to RGB for PIL
            pil_image = Image.fromarray(image[:, :, ::-1])
        else:
            # Grayscale
            pil_image = Image.fromarray(image)
        
        # Resize using PIL
        resized_pil = pil_image.resize(size, Image.LANCZOS)
        
        # Convert back to numpy array
        resized = np.array(resized_pil)
        
        # Convert RGB back to BGR if needed
        if len(resized.shape) == 3:
            resized = resized[:, :, ::-1]
        
        return resized

def _canny_edge_detection(image: np.ndarray, threshold1: float = 50, threshold2: float = 150) -> np.ndarray:
    """Canny edge detection - fallback if cv2.Canny doesn't exist"""
    if hasattr(cv2, 'Canny'):
        return cv2.Canny(image, int(threshold1), int(threshold2))
    else:
        # Fallback: Simplified edge detection using Sobel operator
        # This is a basic implementation - for full Canny accuracy, OpenCV is preferred
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            gray = _bgr_to_gray(image)
        else:
            gray = image
        
        # Apply Gaussian blur (simplified - using basic smoothing)
        # In a real Canny implementation, this would be a proper Gaussian kernel
        blurred = None
        try:
            from scipy import ndimage
            blurred = ndimage.gaussian_filter(gray.astype(np.float32), sigma=1.0).astype(np.uint8)
        except (ImportError, AttributeError, RuntimeError, OSError, Exception) as e:
            # If scipy is not available or broken, use simple box filter
            # Catch all exceptions since scipy might be installed but broken
            pass
        
        if blurred is None:
            # Use simple box filter (average blur) as fallback
            # This is a basic 3x3 averaging kernel using vectorized operations
            gray_float = gray.astype(np.float32)
            h, w = gray.shape
            
            # Pad image with edge values for proper convolution
            padded = np.pad(gray_float, ((1, 1), (1, 1)), mode='edge')
            
            # Apply 3x3 box filter using vectorized operations
            # Sum over 3x3 neighborhoods
            blurred = np.zeros_like(gray_float)
            for i in range(h):
                for j in range(w):
                    blurred[i, j] = np.mean(padded[i:i+3, j:j+3])
            
            blurred = blurred.astype(np.uint8)
        
        # Sobel operators for gradient detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Calculate gradients using vectorized operations
        blurred_float = blurred.astype(np.float32)
        h, w = blurred.shape
        
        # Pad image for convolution
        padded = np.pad(blurred_float, ((1, 1), (1, 1)), mode='edge')
        
        # Calculate gradients using convolution
        gx = np.zeros_like(blurred_float)
        gy = np.zeros_like(blurred_float)
        
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                region = padded[i-1:i+2, j-1:j+2]
                gx[i-1, j-1] = np.sum(region * sobel_x)
                gy[i-1, j-1] = np.sum(region * sobel_y)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Apply double threshold (simplified version)
        edges = np.zeros_like(magnitude, dtype=np.uint8)
        strong_edges = magnitude > threshold2
        weak_edges = (magnitude > threshold1) & (magnitude <= threshold2)
        
        edges[strong_edges] = 255
        
        # Hysteresis: connect weak edges to strong edges (simplified)
        # Use convolution to find weak edges adjacent to strong edges
        if np.any(weak_edges):
            # Create a kernel to check 8-neighborhood
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0  # Don't count center pixel
            # Convolve strong edges to find neighbors
            strong_padded = np.pad(strong_edges.astype(np.uint8), 1, mode='constant')
            neighbors = np.zeros_like(weak_edges)
            for i in range(1, h + 1):
                for j in range(1, w + 1):
                    neighbors[i-1, j-1] = np.sum(strong_padded[i-1:i+2, j-1:j+2] * kernel) > 0
            
            # Connect weak edges that have strong edge neighbors
            edges[weak_edges & neighbors] = 255
        
        return edges

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
        
        # Template matcher for image detection
        self.template_matcher = TemplateMatcher()
        
        # Region configurations (loaded from JSON)
        self.regions = {}
        self.device_id = None  # Will be set when device is connected
        
        
        # Game state detection patterns
        self.state_text_patterns = {
            'in_game': ['character', 'level', 'exp', 'hp', 'mp', 'attack', 'defense'],
            'main_menu': ['start', 'character', 'settings', 'exit', 'menu'],
            'loading_screen': ['loading', 'please wait', 'connecting'],
            'character_select': ['select character', 'create character', 'character'],
            'inventory': ['inventory', 'items', 'equipment'],
            'quest': ['quest', 'mission', 'objective'],
            'shop': ['shop', 'buy', 'sell', 'npc'],
            'dialogue': ['next', 'continue', 'skip', 'close']
        }
    
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
    
    def detect_game_state(self, skip_tap_screen_check: bool = False, previous_state: str = None) -> Dict:
        """
        Detect current game state using screenshot analysis, image matching, and OCR
        
        Args:
            skip_tap_screen_check: If True, skip tap screen detection (for when already in-game)
            previous_state: Previous game state - if in-game, skip tap screen detection
        
        Returns:
            Dictionary containing game state information
        """
        screenshot = None
        try:
            screenshot = self.adb.take_screenshot()
            if screenshot is None:
                return {
                    'status': 'error', 
                    'message': 'Could not take screenshot',
                    'screenshot_taken': False
                }
            
            # Store screen size before processing (screenshot will be deleted)
            screen_size = screenshot.shape[:2]
            
            # Never check for "Tap screen" if we're already in-game or if explicitly skipped
            # Check previous state first - if in-game, skip tap screen detection
            is_in_game_state = previous_state in ['in_game', 'in_game_with_ui']
            skip_check = skip_tap_screen_check or is_in_game_state
            
            tap_screen_detected = False
            if not skip_check:
                tap_screen_detected = self._detect_tap_screen_text(screenshot)
            else:
                logger.debug("Skipping tap screen detection (already in-game or explicitly skipped)")
            
            # Basic game state detection
            state = {
                'status': 'select_server' if tap_screen_detected else 'unknown',
                'screenshot_taken': True,
                'screen_size': screen_size,
                'timestamp': self._get_current_timestamp(),
                'message': 'Screenshot captured successfully',
                'tap_screen_detected': tap_screen_detected
            }
            
            # If "Tap screen" is detected, we're in server selection state
            if tap_screen_detected:
                logger.info("'Tap screen' text detected - game is in server selection state")
                return state
            
            # Enhanced state detection using images and text
            detected_state = self._detect_game_state_enhanced(screenshot)
            if detected_state:
                state['status'] = detected_state
                logger.info(f"Detected game state: {detected_state}")
            else:
                # Fallback to basic analysis
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
        finally:
            # Always release screenshot after processing
            if screenshot is not None:
                del screenshot
                # Force garbage collection to immediately free memory
                import gc
                gc.collect(0)  # Quick collection of generation 0
                # Clear GPU cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
    
    def _detect_tap_screen_text(self, screenshot: np.ndarray) -> bool:
        """
        Detect "Tap screen" text on screenshot using Tesseract OCR
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            True if "Tap screen" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader (lightweight, ~100MB vs ~2GB for EasyOCR)
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                logger.debug("Tesseract OCR not available for 'Tap screen' detection")
                return False
            
            # Process image at full device resolution (no downsampling)
            # This ensures coordinates match device dimensions exactly
            results = None
            try:
                # Perform OCR on full resolution screenshot (Tesseract works with BGR directly)
                results = ocr_reader.readtext(screenshot, detail=1, paragraph=False)
                
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
                
                found = False
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Check if any variation matches
                    for variation in tap_screen_variations:
                        if variation in text_lower:
                            logger.info(f"Detected 'Tap screen' text: '{text}' (confidence: {confidence:.3f})")
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    logger.debug("'Tap screen' text not detected in screenshot")
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting 'Tap screen' text: {e}")
            return False
    
    def _detect_select_character_text(self, screenshot: np.ndarray) -> bool:
        """
        Detect "Select Character" text on screenshot using Tesseract OCR
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            True if "Select Character" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                logger.debug("Tesseract OCR not available for 'Select Character' detection")
                return False
            
            results = None
            try:
                # Perform OCR on full resolution screenshot
                results = ocr_reader.readtext(screenshot, detail=1, paragraph=False)
                
                # Search for "Select Character" or variations
                select_character_variations = [
                    'select character',
                    'select a character',
                    'character select',
                    'choose character',
                    'character',
                    'select'
                ]
                
                found = False
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Check if any variation matches
                    for variation in select_character_variations:
                        if variation in text_lower:
                            logger.info(f"Detected 'Select Character' text: '{text}' (confidence: {confidence:.3f})")
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    logger.debug("'Select Character' text not detected in screenshot")
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting 'Select Character' text: {e}")
            return False
    
    def detect_select_character_and_enter_button(self, screenshot: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Detect both "Select Character" text AND enter button in screenshot
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            Tuple of (both_detected, enter_button_position)
            - both_detected: True if both "Select Character" text and enter button are detected
            - enter_button_position: (x, y) position of enter button if found, None otherwise
        """
        try:
            # Check for "Select Character" text
            select_character_detected = self._detect_select_character_text(screenshot)
            
            # Check for enter button using template matching
            enter_button_position = None
            try:
                enter_result = self.template_matcher.find_template(screenshot, "enter_button.png", multi_scale=True)
                if enter_result:
                    x, y, confidence = enter_result
                    enter_button_position = (x, y)
                    logger.info(f"Enter button detected at ({x}, {y}) with confidence {confidence:.3f}")
            except Exception as e:
                logger.debug(f"Error detecting enter button: {e}")
            
            # Both must be detected
            both_detected = select_character_detected and (enter_button_position is not None)
            
            if both_detected:
                logger.info("Both 'Select Character' text and enter button detected - game is in character selection state")
            else:
                if not select_character_detected:
                    logger.debug("'Select Character' text not detected")
                if enter_button_position is None:
                    logger.debug("Enter button not detected")
            
            return both_detected, enter_button_position
            
        except Exception as e:
            logger.debug(f"Error detecting select character and enter button: {e}")
            return False, None
    
    def _detect_select_server_button(self, screenshot: np.ndarray) -> bool:
        """
        Detect "Select Server" button using template matching
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            True if "Select Server" button is detected, False otherwise
        """
        try:
            # Use the actual template file name from assets/templates
            result = self.template_matcher.find_template(screenshot, "select_server_button.png", multi_scale=True, confidence=0.7)
            if result:
                x, y, confidence = result
                logger.info(f"Select Server button detected at ({x}, {y}) with confidence {confidence:.3f}")
                return True
            
            return False
        except Exception as e:
            logger.debug(f"Error detecting Select Server button: {e}")
            return False
    
    def _detect_fight_button(self, screenshot: np.ndarray) -> bool:
        """
        Detect "Short" button using template matching within the short_button_region
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            True if "Short" button is detected, False otherwise
        """
        try:
            # Ensure regions are loaded
            if not self.regions and self.adb and self.adb.device_id:
                self._load_regions(self.adb.device_id)
            
            # Get short_button_region
            short_button_region = self._get_region('short_button_region')
            if not short_button_region:
                logger.warning("short_button_region not configured - searching full screenshot (this may be slower and less accurate)")
                # Fallback to full screenshot if region not configured
                result = self.template_matcher.find_template(screenshot, "short_button.png", multi_scale=True, confidence=0.7)
            else:
                x1, y1, x2, y2 = short_button_region
                logger.debug(f"Using short_button_region: ({x1}, {y1}, {x2}, {y2})")
                
                # Ensure coordinates are within screenshot bounds
                x1 = max(0, min(x1, screenshot.shape[1]))
                y1 = max(0, min(y1, screenshot.shape[0]))
                x2 = max(0, min(x2, screenshot.shape[1]))
                y2 = max(0, min(y2, screenshot.shape[0]))
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid short_button_region coordinates: ({x1}, {y1}, {x2}, {y2})")
                    return False
                
                # Convert region to (x, y, width, height) format for template matcher
                region_width = x2 - x1
                region_height = y2 - y1
                search_region = (x1, y1, region_width, region_height)
                logger.debug(f"Searching for short_button.png in region: ({x1}, {y1}, {region_width}, {region_height})")
                
                # Search for short_button.png within the region
                result = self.template_matcher.find_template(
                    screenshot,
                    "short_button.png",
                    multi_scale=True,
                    confidence=0.7,
                    region=search_region
                )
            
            if result:
                x, y, confidence = result
                logger.info(f"Short button detected at ({x}, {y}) with confidence {confidence:.3f}")
                return True
            
            return False
        except Exception as e:
            logger.debug(f"Error detecting Short button: {e}")
            return False
    
    def _detect_character_text(self, screenshot: np.ndarray) -> bool:
        """
        Detect "Character" text on screenshot using Tesseract OCR
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            True if "Character" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                logger.debug("Tesseract OCR not available for 'Character' text detection")
                return False
            
            results = None
            try:
                # Perform OCR on full resolution screenshot
                results = ocr_reader.readtext(screenshot, detail=1, paragraph=False)
                
                # Search for "Character" text (case-insensitive)
                character_keywords = ['character', 'characters']
                
                found = False
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Check if "character" is in the text
                    for keyword in character_keywords:
                        if keyword in text_lower:
                            logger.info(f"Detected 'Character' text: '{text}' (confidence: {confidence:.3f})")
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    logger.debug("'Character' text not detected in screenshot")
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting 'Character' text: {e}")
            return False
    
    def _detect_enter_button(self, screenshot: np.ndarray) -> bool:
        """
        Detect enter_button.png template in screenshot
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            True if enter_button.png is detected, False otherwise
        """
        try:
            enter_result = self.template_matcher.find_template(screenshot, "enter_button.png", multi_scale=True)
            if enter_result:
                x, y, confidence = enter_result
                logger.info(f"Detected enter_button.png at ({x}, {y}) with confidence {confidence:.3f}")
                return True
            else:
                logger.debug("enter_button.png not detected in screenshot")
                return False
        except Exception as e:
            logger.debug(f"Error detecting enter button: {e}")
            return False
    
    # Debug versions of detection methods that return detailed information
    def _detect_tap_screen_text_debug(self, screenshot: np.ndarray) -> dict:
        """Debug version that returns detailed detection info"""
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return {'detected': False, 'detections': []}
            
            results = None
            detections = []
            try:
                results = ocr_reader.readtext(screenshot, detail=1, paragraph=False)
                tap_variations = ['tap screen', 'tap to screen', 'tap the screen', 'tap screen to', 'tap', 'tap to start', 'tap to begin']
                
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    for variation in tap_variations:
                        if variation in text_lower:
                            x1, y1 = bbox[0]
                            x2, y2 = bbox[2]
                            detections.append({
                                'text': text,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence)
                            })
                            break
                return {'detected': len(detections) > 0, 'detections': detections}
            finally:
                if results is not None:
                    del results
        except Exception as e:
            logger.debug(f"Error in debug tap detection: {e}")
            return {'detected': False, 'detections': []}
    
    def _detect_character_text_debug(self, screenshot: np.ndarray) -> dict:
        """Debug version that returns detailed detection info"""
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return {'detected': False, 'detections': []}
            
            results = None
            detections = []
            try:
                results = ocr_reader.readtext(screenshot, detail=1, paragraph=False)
                character_keywords = ['character', 'characters']
                
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    for keyword in character_keywords:
                        if keyword in text_lower:
                            x1, y1 = bbox[0]
                            x2, y2 = bbox[2]
                            detections.append({
                                'text': text,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence)
                            })
                            break
                return {'detected': len(detections) > 0, 'detections': detections}
            finally:
                if results is not None:
                    del results
        except Exception as e:
            logger.debug(f"Error in debug character detection: {e}")
            return {'detected': False, 'detections': []}
    
    def _detect_select_server_button_debug(self, screenshot: np.ndarray) -> dict:
        """Debug version that returns detailed detection info, including best match even if below threshold"""
        try:
            # First try with normal threshold
            result = self.template_matcher.find_template(screenshot, "select_server_button.png", multi_scale=True)
            if result:
                x, y, confidence = result
                template = self.template_matcher.load_template("select_server_button.png")
                size = (template.shape[1], template.shape[0]) if template is not None else (100, 100)
                return {'detected': True, 'position': (int(x), int(y)), 'confidence': float(confidence), 'size': size}
            
            # If not found, try with lower threshold to find best match
            logger.info("Template not found with normal threshold, trying lower threshold for debugging...")
            best_match = self.template_matcher.find_template_best_match(screenshot, "select_server_button.png", multi_scale=True)
            if best_match:
                x, y, confidence = best_match
                template = self.template_matcher.load_template("select_server_button.png")
                size = (template.shape[1], template.shape[0]) if template is not None else (100, 100)
                logger.warning(f"Best match found (below threshold): position=({x}, {y}), confidence={confidence:.3f}")
                return {'detected': False, 'position': (int(x), int(y)), 'confidence': float(confidence), 'size': size, 'below_threshold': True}
            
            return {'detected': False, 'position': None}
        except Exception as e:
            logger.debug(f"Error in debug select server button detection: {e}")
            return {'detected': False, 'position': None}
    
    def _detect_enter_button_debug(self, screenshot: np.ndarray) -> dict:
        """Debug version that returns detailed detection info, including best match even if below threshold"""
        try:
            result = self.template_matcher.find_template(screenshot, "enter_button.png", multi_scale=True)
            if result:
                x, y, confidence = result
                template = self.template_matcher.load_template("enter_button.png")
                size = (template.shape[1], template.shape[0]) if template is not None else (100, 100)
                return {'detected': True, 'position': (int(x), int(y)), 'confidence': float(confidence), 'size': size}
            
            # Try with lower threshold
            logger.info("Enter button not found with normal threshold, trying lower threshold for debugging...")
            best_match = self.template_matcher.find_template_best_match(screenshot, "enter_button.png", multi_scale=True)
            if best_match:
                x, y, confidence = best_match
                template = self.template_matcher.load_template("enter_button.png")
                size = (template.shape[1], template.shape[0]) if template is not None else (100, 100)
                logger.warning(f"Best match found (below threshold): position=({x}, {y}), confidence={confidence:.3f}")
                return {'detected': False, 'position': (int(x), int(y)), 'confidence': float(confidence), 'size': size, 'below_threshold': True}
            
            return {'detected': False, 'position': None}
        except Exception as e:
            logger.debug(f"Error in debug enter button detection: {e}")
            return {'detected': False, 'position': None}
    
    def _detect_fight_button_debug(self, screenshot: np.ndarray) -> dict:
        """Debug version that returns detailed detection info, including best match even if below threshold"""
        try:
            # Ensure regions are loaded
            if not self.regions and self.adb and self.adb.device_id:
                self._load_regions(self.adb.device_id)
            
            # Get short_button_region
            short_button_region = self._get_region('short_button_region')
            search_region = None
            
            if not short_button_region:
                logger.warning("short_button_region not configured - searching full screenshot (this may be slower and less accurate)")
            else:
                x1, y1, x2, y2 = short_button_region
                logger.debug(f"Using short_button_region: ({x1}, {y1}, {x2}, {y2})")
                
                # Ensure coordinates are within screenshot bounds
                x1 = max(0, min(x1, screenshot.shape[1]))
                y1 = max(0, min(y1, screenshot.shape[0]))
                x2 = max(0, min(x2, screenshot.shape[1]))
                y2 = max(0, min(y2, screenshot.shape[0]))
                
                if x2 > x1 and y2 > y1:
                    # Convert region to (x, y, width, height) format for template matcher
                    region_width = x2 - x1
                    region_height = y2 - y1
                    search_region = (x1, y1, region_width, region_height)
                    logger.debug(f"Searching for short_button.png in region: ({x1}, {y1}, {region_width}, {region_height})")
                else:
                    logger.warning(f"Invalid short_button_region coordinates: ({x1}, {y1}, {x2}, {y2})")
            
            # Search for short_button.png within the region (or full screenshot if region not configured)
            result = self.template_matcher.find_template(
                screenshot, 
                "short_button.png", 
                multi_scale=True, 
                confidence=0.7,
                region=search_region
            )
            
            if result:
                x, y, confidence = result
                template = self.template_matcher.load_template("short_button.png")
                size = (template.shape[1], template.shape[0]) if template is not None else (100, 100)
                logger.info(f"Short button detected in {'region' if search_region else 'full screenshot'} at ({x}, {y}) with confidence {confidence:.3f}")
                return {'detected': True, 'position': (int(x), int(y)), 'confidence': float(confidence), 'size': size}
            
            # Try with lower threshold
            logger.info("Short button not found with normal threshold, trying lower threshold for debugging...")
            best_match = self.template_matcher.find_template_best_match(
                screenshot, 
                "short_button.png", 
                multi_scale=True,
                region=search_region
            )
            if best_match:
                x, y, confidence = best_match
                template = self.template_matcher.load_template("short_button.png")
                size = (template.shape[1], template.shape[0]) if template is not None else (100, 100)
                logger.warning(f"Best match found (below threshold): position=({x}, {y}), confidence={confidence:.3f} in {'region' if search_region else 'full screenshot'}")
                return {'detected': False, 'position': (int(x), int(y)), 'confidence': float(confidence), 'size': size, 'below_threshold': True}
            
            return {'detected': False, 'position': None}
        except Exception as e:
            logger.debug(f"Error in debug short button detection: {e}")
            return {'detected': False, 'position': None}
    
    def _detect_quest_button_only(self, screenshot: np.ndarray) -> bool:
        """
        Detect quest button in agent_quest_region (detection only, no tapping)
        
        Args:
            screenshot: OpenCV image array (full screenshot)
            
        Returns:
            True if button was detected, False otherwise
        """
        try:
            # Get agent_quest_region
            agent_quest_region = self._get_region('agent_quest_region')
            if not agent_quest_region:
                logger.debug("agent_quest_region not configured - cannot detect quest button")
                return False
            
            x1, y1, x2, y2 = agent_quest_region
            
            # Ensure coordinates are within screenshot bounds
            x1 = max(0, min(x1, screenshot.shape[1]))
            y1 = max(0, min(y1, screenshot.shape[0]))
            x2 = max(0, min(x2, screenshot.shape[1]))
            y2 = max(0, min(y2, screenshot.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            # Convert region to (x, y, width, height) format for template matcher
            region_width = x2 - x1
            region_height = y2 - y1
            search_region = (x1, y1, region_width, region_height)
            
            # Search for quest_button.png within the region
            result = self.template_matcher.find_template(
                screenshot, 
                "quest_button.png", 
                multi_scale=True, 
                confidence=0.7,
                region=search_region
            )
            
            return result is not None
            
        except Exception as e:
            logger.debug(f"Error detecting quest button: {e}")
            return False
    
    def detect_and_tap_unknown_state_buttons(self, screenshot: np.ndarray) -> bool:
        """
        Detect and tap buttons that appear in unknown state:
        - confirm_button_1.png
        - claim_reward_button.png
        - accept_button.png
        
        Args:
            screenshot: OpenCV image array (full screenshot)
            
        Returns:
            True if any button was detected and tapped, False otherwise
        """
        try:
            # List of buttons to check in order
            buttons_to_check = [
                "confirm_button_1.png",
                "claim_reward_button.png",
                "accept_button.png"
            ]
            
            logger.info("Checking for unknown state buttons (confirm_button_1, claim_reward_button, accept_button)...")
            
            # Search for each button in the full screenshot (no specific region)
            for button_template in buttons_to_check:
                result = self.template_matcher.find_template(
                    screenshot,
                    button_template,
                    multi_scale=True,
                    confidence=0.7
                )
                
                if result:
                    x, y, confidence = result
                    button_name = button_template.replace(".png", "")
                    logger.info(f"[OK] {button_name} detected at ({x}, {y}) with confidence {confidence:.3f}")
                    
                    # Tap the button
                    if self.adb and hasattr(self.adb, 'tap'):
                        success = self.adb.tap(int(x), int(y))
                        if success:
                            logger.info(f"[OK] Successfully tapped {button_name} at ({x}, {y})")
                            return True
                        else:
                            logger.warning(f"[FAIL] Failed to tap {button_name} at ({x}, {y})")
                            # Continue to next button if tap failed
                            continue
                    else:
                        logger.warning("ADB manager not available or tap method not found")
                        return False
            
            logger.debug("No unknown state buttons detected (confirm_button_1, claim_reward_button, accept_button)")
            return False
            
        except Exception as e:
            logger.error(f"Error detecting/tapping unknown state buttons: {e}")
            return False
    
    def detect_and_tap_agent_quest_button(self, screenshot: np.ndarray) -> bool:
        """
        Detect quest button in agent_quest_region, tap it, then wait for and tap confirm button
        
        Args:
            screenshot: OpenCV image array (full screenshot)
            
        Returns:
            True if quest button was detected and tapped (and confirm button was tapped), False otherwise
        """
        try:
            # Get agent_quest_region
            agent_quest_region = self._get_region('agent_quest_region')
            if not agent_quest_region:
                logger.debug("agent_quest_region not configured - cannot detect quest button")
                return False
            
            x1, y1, x2, y2 = agent_quest_region
            logger.info(f"Checking for quest button in region: ({x1}, {y1}, {x2}, {y2})")
            
            # Ensure coordinates are within screenshot bounds
            x1 = max(0, min(x1, screenshot.shape[1]))
            y1 = max(0, min(y1, screenshot.shape[0]))
            x2 = max(0, min(x2, screenshot.shape[1]))
            y2 = max(0, min(y2, screenshot.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid agent_quest_region coordinates: ({x1}, {y1}, {x2}, {y2})")
                return False
            
            # Convert region to (x, y, width, height) format for template matcher
            region_width = x2 - x1
            region_height = y2 - y1
            search_region = (x1, y1, region_width, region_height)
            logger.debug(f"Searching for quest_button.png in region: ({x1}, {y1}, {region_width}, {region_height})")
            
            # Search for quest_button.png within the region
            result = self.template_matcher.find_template(
                screenshot, 
                "quest_button.png", 
                multi_scale=True, 
                confidence=0.7,
                region=search_region
            )
            
            if result:
                x, y, confidence = result
                logger.info(f"[OK] Quest button detected at ({x}, {y}) with confidence {confidence:.3f}")
                
                # Tap the quest button
                if self.adb and hasattr(self.adb, 'tap'):
                    success = self.adb.tap(int(x), int(y))
                    if success:
                        logger.info(f"[OK] Successfully tapped quest button at ({x}, {y})")
                        
                        # After tapping quest button, wait for confirm button and tap it
                        logger.info("Waiting for confirm button to appear...")
                        confirm_tapped = self._wait_and_tap_confirm_button(max_wait_seconds=5.0)
                        if confirm_tapped:
                            logger.info("[OK] Confirm button detected and tapped successfully - setting state to auto_questing")
                            # Return a special value to indicate confirm button was tapped
                            return 'auto_questing'
                        else:
                            logger.warning("[WARN] Quest button tapped but confirm button not found within timeout")
                            return True  # Return True because quest button was tapped successfully
                    else:
                        logger.warning(f"[FAIL] Failed to tap quest button at ({x}, {y})")
                        return False
                else:
                    logger.warning("ADB manager not available or tap method not found")
                    return False
            
            logger.debug("Quest button not found in agent_quest_region")
            return False
            
        except Exception as e:
            logger.error(f"Error detecting/tapping quest button: {e}")
            return False
    
    def _wait_and_tap_confirm_button(self, max_wait_seconds: float = 5.0, check_interval: float = 0.5) -> bool:
        """
        Wait for confirm_button_1.png to appear and tap it
        
        Args:
            max_wait_seconds: Maximum time to wait for confirm button (default: 5 seconds)
            check_interval: Interval between checks in seconds (default: 0.5 seconds)
            
        Returns:
            True if confirm button was detected and tapped, False otherwise
        """
        try:
            import time
            start_time = time.time()
            check_count = 0
            
            logger.debug(f"Waiting for confirm_button_1.png (max wait: {max_wait_seconds}s, check interval: {check_interval}s)")
            
            while (time.time() - start_time) < max_wait_seconds:
                check_count += 1
                
                # Take a new screenshot to check for confirm button
                screenshot = self.adb.take_screenshot()
                if screenshot is None:
                    logger.debug("Could not take screenshot while waiting for confirm button")
                    time.sleep(check_interval)
                    continue
                
                # Search for confirm_button_1.png in the full screenshot
                result = self.template_matcher.find_template(
                    screenshot,
                    "confirm_button_1.png",
                    multi_scale=True,
                    confidence=0.7
                )
                
                if result:
                    x, y, confidence = result
                    elapsed_time = time.time() - start_time
                    logger.info(f"[OK] Confirm button detected at ({x}, {y}) with confidence {confidence:.3f} after {elapsed_time:.2f}s")
                    
                    # Tap the confirm button
                    if self.adb and hasattr(self.adb, 'tap'):
                        success = self.adb.tap(int(x), int(y))
                        if success:
                            logger.info(f"[OK] Successfully tapped confirm button at ({x}, {y})")
                            return True
                        else:
                            logger.warning(f"[FAIL] Failed to tap confirm button at ({x}, {y})")
                            return False
                    else:
                        logger.warning("ADB manager not available or tap method not found")
                        return False
                
                # Wait before next check
                time.sleep(check_interval)
            
            elapsed_time = time.time() - start_time
            logger.warning(f"[TIMEOUT] Confirm button not found after {elapsed_time:.2f}s (checked {check_count} times)")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for/tapping confirm button: {e}")
            return False
    
    def _save_template_images(self, template_names: list):
        """Save template images to debug folder for reference"""
        try:
            import datetime
            import shutil
            debug_dir = Path('debug_screenshots')
            debug_dir.mkdir(exist_ok=True)
            
            templates_dir = self.template_matcher.templates_dir
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            for template_name in template_names:
                template_path = templates_dir / template_name
                if template_path.exists():
                    debug_template_path = debug_dir / f"template_{timestamp}_{template_name}"
                    shutil.copy2(template_path, debug_template_path)
                    logger.info(f"Debug template saved: {debug_template_path}")
        except Exception as e:
            logger.debug(f"Error saving template images: {e}")
    
    def detect_game_state_from_screenshot(self, screenshot: np.ndarray, debug: bool = False) -> str:
        """
        Detect game state from screenshot based on specific element combinations
        
        Basic main loop rules:
        1. If screenshot includes "tap" text AND select_server_button.png → "select_server"
        2. If screenshot includes "Character" text AND enter_button.png → "select_character"
        3. If screenshot includes short_button.png → "playing"
        4. Otherwise → "unknown"
        
        Args:
            screenshot: OpenCV image array
            debug: If True, save screenshot and detailed detection results
            
        Returns:
            Game state string: 
            - 'select_server', 'select_character', 'unknown'
            - 'playing' (in-game state)
        """
        try:
            logger.info("Starting game state detection from screenshot...")
            
            # Collect annotation data for debug images
            annotations = {'templates': [], 'text': []}
            
            # Save screenshot for debugging if requested
            if debug:
                self._save_debug_screenshot(screenshot, "detection_start")
            
            # Rule 1: Check for "tap" text AND select_server_button.png
            logger.info("Rule 1: Checking for 'tap' text and select_server_button.png...")
            tap_result = self._detect_tap_screen_text_debug(screenshot) if debug else self._detect_tap_screen_text(screenshot)
            if debug and isinstance(tap_result, dict):
                tap_detected = tap_result.get('detected', False)
                if tap_result.get('detections'):
                    annotations['text'].extend(tap_result['detections'])
            else:
                tap_detected = tap_result if isinstance(tap_result, bool) else False
            logger.info(f"  - 'tap' text detected: {tap_detected}")
            
            select_server_result = self._detect_select_server_button_debug(screenshot) if debug else self._detect_select_server_button(screenshot)
            if debug and isinstance(select_server_result, dict):
                select_server_button_detected = select_server_result.get('detected', False)
                if select_server_result.get('position'):
                    annotations['templates'].append({
                        'name': 'select_server_button',
                        'position': select_server_result['position'],
                        'confidence': select_server_result.get('confidence', 0),
                        'size': select_server_result.get('size', (100, 100)),
                        'below_threshold': select_server_result.get('below_threshold', False)
                    })
            else:
                select_server_button_detected = select_server_result if isinstance(select_server_result, bool) else False
            logger.info(f"  - select_server_button.png detected: {select_server_button_detected}")
            
            if tap_detected and select_server_button_detected:
                detected_state = 'select_server'
                logger.info(f"Game state detected: {detected_state} (tap text + select_server_button.png)")
                if debug:
                    self._save_debug_screenshot(screenshot, f"detected_{detected_state}", annotations)
                    self._save_template_images(['select_server_button.png'])
                return detected_state
            
            # Rule 2: Check for "Character" text AND enter_button.png
            logger.info("Rule 2: Checking for 'Character' text and enter_button.png...")
            character_result = self._detect_character_text_debug(screenshot) if debug else self._detect_character_text(screenshot)
            if debug and isinstance(character_result, dict):
                character_text_detected = character_result.get('detected', False)
                if character_result.get('detections'):
                    annotations['text'].extend(character_result['detections'])
            else:
                character_text_detected = character_result if isinstance(character_result, bool) else False
            logger.info(f"  - 'Character' text detected: {character_text_detected}")
            
            enter_result = self._detect_enter_button_debug(screenshot) if debug else self._detect_enter_button(screenshot)
            if debug and isinstance(enter_result, dict):
                enter_button_detected = enter_result.get('detected', False)
                if enter_result.get('position'):
                    annotations['templates'].append({
                        'name': 'enter_button',
                        'position': enter_result['position'],
                        'confidence': enter_result.get('confidence', 0),
                        'size': enter_result.get('size', (100, 100)),
                        'below_threshold': enter_result.get('below_threshold', False)
                    })
            else:
                enter_button_detected = enter_result if isinstance(enter_result, bool) else False
            logger.info(f"  - enter_button.png detected: {enter_button_detected}")
            
            if character_text_detected and enter_button_detected:
                detected_state = 'select_character'
                logger.info(f"Game state detected: {detected_state} (Character text + enter_button.png)")
                if debug:
                    self._save_debug_screenshot(screenshot, f"detected_{detected_state}", annotations)
                    self._save_template_images(['enter_button.png'])
                return detected_state
            
            # Rule 3: Check for short_button.png
            logger.info("Rule 3: Checking for short_button.png...")
            fight_result = self._detect_fight_button_debug(screenshot) if debug else self._detect_fight_button(screenshot)
            if debug and isinstance(fight_result, dict):
                fight_button_detected = fight_result.get('detected', False)
                if fight_result.get('position'):
                    annotations['templates'].append({
                        'name': 'short_button',
                        'position': fight_result['position'],
                        'confidence': fight_result.get('confidence', 0),
                        'size': fight_result.get('size', (100, 100)),
                        'below_threshold': fight_result.get('below_threshold', False)
                    })
            else:
                fight_button_detected = fight_result if isinstance(fight_result, bool) else False
            logger.info(f"  - short_button.png detected: {fight_button_detected}")
            
            if fight_button_detected:
                logger.info("Game state detected: 'playing' (short_button.png found)")
                
                if debug:
                    self._save_debug_screenshot(screenshot, "detected_playing", annotations)
                    self._save_template_images(['short_button.png'])
                
                return 'playing'
            
            # No matching state found
            detected_state = 'unknown'
            logger.info("Game state not determined from screenshot elements - returning 'unknown'")
            logger.info(f"  Detection Summary:")
            logger.info(f"    - tap text: {tap_detected}")
            logger.info(f"    - select_server_button: {select_server_button_detected}")
            logger.info(f"    - character text: {character_text_detected}")
            logger.info(f"    - enter_button: {enter_button_detected}")
            logger.info(f"    - short_button: {fight_button_detected}")
            logger.info(f"  Result: {detected_state}")
            
            if debug:
                self._save_debug_screenshot(screenshot, f"detected_{detected_state}", annotations)
                self._save_template_images(['select_server_button.png', 'enter_button.png', 'short_button.png'])
                self._save_detection_results({
                    'detected_state': detected_state,
                    'tap_detected': tap_detected,
                    'select_server_button_detected': select_server_button_detected,
                    'character_text_detected': character_text_detected,
                    'enter_button_detected': enter_button_detected,
                    'short_button_detected': fight_button_detected,
                    'annotations': annotations
                })
            
            return detected_state
            
        except Exception as e:
            logger.error(f"Error detecting game state from screenshot: {e}", exc_info=True)
            return 'unknown'
    
    def _save_debug_screenshot(self, screenshot: np.ndarray, suffix: str, annotations: dict = None):
        """
        Save screenshot for debugging purposes with optional annotations
        
        Args:
            screenshot: OpenCV image array
            suffix: Suffix for filename
            annotations: Dictionary with annotation data:
                - 'templates': List of {'name': str, 'position': (x, y), 'confidence': float}
                - 'text': List of {'text': str, 'bbox': (x1, y1, x2, y2), 'confidence': float}
        """
        try:
            import datetime
            debug_dir = Path('debug_screenshots')
            debug_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Create annotated screenshot if annotations provided
            annotated_screenshot = screenshot.copy() if annotations else screenshot
            
            if annotations:
                annotated_screenshot = self._draw_debug_annotations(annotated_screenshot, annotations)
            
            filename = debug_dir / f"screenshot_{timestamp}_{suffix}.png"
            
            # Try to save using cv2.imwrite, fallback to PIL
            try:
                if hasattr(cv2, 'imwrite'):
                    cv2.imwrite(str(filename), annotated_screenshot)
                else:
                    from PIL import Image, ImageDraw, ImageFont
                    # Convert BGR to RGB for PIL
                    if len(annotated_screenshot.shape) == 3:
                        rgb_image = annotated_screenshot[:, :, ::-1]
                        pil_image = Image.fromarray(rgb_image)
                    else:
                        pil_image = Image.fromarray(annotated_screenshot)
                    
                    # Draw annotations using PIL if cv2 not available
                    if annotations:
                        pil_image = self._draw_debug_annotations_pil(pil_image, annotations)
                    
                    pil_image.save(filename)
                logger.info(f"Debug screenshot saved: {filename}")
            except Exception as e:
                logger.warning(f"Failed to save debug screenshot: {e}")
        except Exception as e:
            logger.debug(f"Error saving debug screenshot: {e}")
    
    def _draw_debug_annotations(self, screenshot: np.ndarray, annotations: dict) -> np.ndarray:
        """Draw debug annotations (boxes, labels) on screenshot using OpenCV"""
        try:
            annotated = screenshot.copy()
            
            # Draw template detections
            if 'templates' in annotations:
                for template_info in annotations['templates']:
                    name = template_info.get('name', 'template')
                    position = template_info.get('position')
                    confidence = template_info.get('confidence', 0)
                    template_size = template_info.get('size', (100, 100))  # Default size if not provided
                    below_threshold = template_info.get('below_threshold', False)
                    
                    if position:
                        x, y = position
                        w, h = template_size
                        
                        # Draw rectangle - green if above threshold, yellow if below
                        color = (0, 255, 255) if below_threshold else (0, 255, 0)  # Yellow or green
                        if hasattr(cv2, 'rectangle'):
                            cv2.rectangle(annotated, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
                            # Draw label
                            if hasattr(cv2, 'putText'):
                                threshold_status = " (BELOW THRESHOLD)" if below_threshold else ""
                                label = f"{name} ({confidence:.3f}){threshold_status}"
                                font = cv2.FONT_HERSHEY_SIMPLEX if hasattr(cv2, 'FONT_HERSHEY_SIMPLEX') else 0
                                cv2.putText(annotated, label, (x - w//2, y - h//2 - 10), 
                                          font, 0.7, color, 2)
                        else:
                            # Fallback: draw using PIL later
                            pass
            
            # Draw text detections
            if 'text' in annotations:
                for text_info in annotations['text']:
                    text = text_info.get('text', '')
                    bbox = text_info.get('bbox')
                    confidence = text_info.get('confidence', 0)
                    
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        
                        # Draw rectangle
                        if hasattr(cv2, 'rectangle'):
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            # Draw label
                            if hasattr(cv2, 'putText'):
                                label = f"{text} ({confidence:.2f})"
                                font = cv2.FONT_HERSHEY_SIMPLEX if hasattr(cv2, 'FONT_HERSHEY_SIMPLEX') else 0
                                cv2.putText(annotated, label, (x1, y1 - 10), 
                                          font, 0.7, (255, 0, 0), 2)
                        else:
                            # Fallback: draw using PIL later
                            pass
            
            return annotated
        except Exception as e:
            logger.debug(f"Error drawing debug annotations: {e}")
            return screenshot
    
    def _draw_debug_annotations_pil(self, pil_image, annotations: dict):
        """Draw debug annotations using PIL (fallback when OpenCV not available)"""
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(pil_image)
            
            # Draw template detections
            if 'templates' in annotations:
                for template_info in annotations['templates']:
                    name = template_info.get('name', 'template')
                    position = template_info.get('position')
                    confidence = template_info.get('confidence', 0)
                    template_size = template_info.get('size', (100, 100))
                    below_threshold = template_info.get('below_threshold', False)
                    
                    if position:
                        x, y = position
                        w, h = template_size
                        # Draw rectangle - yellow if below threshold, green if above
                        color = 'yellow' if below_threshold else 'green'
                        draw.rectangle([x - w//2, y - h//2, x + w//2, y + h//2], 
                                      outline=color, width=2)
                        # Draw label
                        threshold_status = " (BELOW THRESHOLD)" if below_threshold else ""
                        label = f"{name} ({confidence:.3f}){threshold_status}"
                        draw.text((x - w//2, y - h//2 - 15), label, fill=color)
            
            # Draw text detections
            if 'text' in annotations:
                for text_info in annotations['text']:
                    text = text_info.get('text', '')
                    bbox = text_info.get('bbox')
                    confidence = text_info.get('confidence', 0)
                    
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        # Draw rectangle
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                        # Draw label
                        label = f"{text} ({confidence:.2f})"
                        draw.text((x1, y1 - 15), label, fill='red')
            
            return pil_image
        except Exception as e:
            logger.debug(f"Error drawing debug annotations with PIL: {e}")
            return pil_image
    
    def _save_detection_results(self, results: dict):
        """Save detection results to a JSON file for debugging"""
        try:
            import json
            import datetime
            debug_dir = Path('debug_screenshots')
            debug_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = debug_dir / f"detection_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detection results saved: {filename}")
        except Exception as e:
            logger.debug(f"Error saving detection results: {e}")
    
    def get_detection_results(self, screenshot: np.ndarray) -> dict:
        """
        Get detailed detection results for debugging
        
        Args:
            screenshot: OpenCV image array
            
        Returns:
            Dictionary with detailed detection results
        """
        try:
            results = {
                'screenshot_shape': screenshot.shape if screenshot is not None else None,
                'detections': {}
            }
            
            # Rule 1
            tap_detected = self._detect_tap_screen_text(screenshot)
            select_server_button_detected = self._detect_select_server_button(screenshot)
            results['detections']['rule1_select_server'] = {
                'tap_text': tap_detected,
                'select_server_button': select_server_button_detected,
                'match': tap_detected and select_server_button_detected
            }
            
            # Rule 2
            character_text_detected = self._detect_character_text(screenshot)
            enter_button_detected = self._detect_enter_button(screenshot)
            results['detections']['rule2_select_character'] = {
                'character_text': character_text_detected,
                'enter_button': enter_button_detected,
                'match': character_text_detected and enter_button_detected
            }
            
            # Rule 3
            fight_button_detected = self._detect_fight_button(screenshot)
            results['detections']['rule3_playing'] = {
                'short_button': fight_button_detected,
                'match': fight_button_detected
            }
            
            # Final state
            detected_state = 'unknown'
            if results['detections']['rule1_select_server']['match']:
                detected_state = 'select_server'
            elif results['detections']['rule2_select_character']['match']:
                detected_state = 'select_character'
            elif results['detections']['rule3_playing']['match']:
                # Playing state detected - check for agent quest button (detection only, no tapping)
                detected_state = 'playing'
                quest_button_detected = self._detect_quest_button_only(screenshot)
                results['detections']['rule3_playing']['quest_button'] = {
                    'detected': quest_button_detected
                }
            
            results['detected_state'] = detected_state
            
            return results
        except Exception as e:
            logger.error(f"Error getting detection results: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _load_regions(self, device_id: str = None):
        """Load region configurations from JSON file"""
        try:
            # Try to load device-specific region file
            if device_id:
                self.device_id = device_id
                # Replace both colons and dots to match file naming convention
                # e.g., "127.0.0.1:5555" -> "127_0_0_1_5555"
                device_filename = device_id.replace(':', '_').replace('.', '_')
                region_file = Path('config') / f'regions_{device_filename}.json'
                logger.debug(f"Attempting to load regions from: {region_file}")
            else:
                region_file = Path('config') / 'regions.json'
            
            if region_file.exists():
                with open(region_file, 'r') as f:
                    self.regions = json.load(f)
                logger.info(f"Loaded {len(self.regions)} region types from {region_file}")
                # Log available region names for debugging
                if self.regions:
                    logger.debug(f"Available regions: {list(self.regions.keys())}")
            else:
                logger.warning(f"Region file not found: {region_file}")
                self.regions = {}
        except Exception as e:
            logger.warning(f"Failed to load regions from {region_file if 'region_file' in locals() else 'unknown file'}: {e}")
            self.regions = {}
    
    def _get_region(self, region_name: str) -> Optional[Tuple[int, int, int, int]]:
        """Get region coordinates for a named region"""
        if not self.regions and self.adb.device_id:
            self._load_regions(self.adb.device_id)
        
        if region_name in self.regions and self.regions[region_name]:
            region_list = self.regions[region_name]
            if region_list and len(region_list) > 0:
                region = region_list[0]
                if len(region) == 4:
                    return tuple(region)
        return None
    
    def _detect_game_state_enhanced(self, screenshot: np.ndarray) -> Optional[str]:
        """
        Enhanced game state detection using image templates, OCR text, and region analysis
        
        Args:
            screenshot: Full resolution screenshot
            
        Returns:
            Detected game state string or None
        """
        try:
            # Load regions if not loaded
            if not self.regions and self.adb.device_id:
                self._load_regions(self.adb.device_id)
            
            # 1. Check for UI elements using template matching
            ui_elements_found = self._detect_ui_elements(screenshot)
            
            # 2. Check for text patterns using OCR
            text_based_state = self._detect_state_from_text(screenshot)
            
            # 3. Check for specific regions (health bar, control buttons, etc.)
            region_based_state = self._detect_state_from_regions(screenshot)
            
            # 4. Combine evidence to determine state
            state = self._combine_state_evidence(ui_elements_found, text_based_state, region_based_state)
            
            return state
            
        except Exception as e:
            logger.debug(f"Error in enhanced state detection: {e}")
            return None
    
    def _detect_ui_elements(self, screenshot: np.ndarray) -> Dict[str, bool]:
        """
        Detect UI elements using template matching
        
        Returns:
            Dictionary of detected UI elements
        """
        detected = {
            'health_bar': False,
            'control_buttons': False,
            'quest_panel': False,
            'inventory': False,
            'menu': False
        }
        
        try:
            # Check for health bar region (indicates in-game)
            health_bar_region = self._get_region('health_bar')
            if health_bar_region:
                x1, y1, x2, y2 = health_bar_region
                if x2 > x1 and y2 > y1:
                    # Check if region has content (not empty)
                    health_area = screenshot[y1:y2, x1:x2]
                    if health_area.size > 0:
                        # Check if health bar is visible (has color variation)
                        gray = _bgr_to_gray(health_area)
                        std_dev = np.std(gray)
                        if std_dev > 10:  # Has variation (not solid color)
                            detected['health_bar'] = True
            
            # Check for control buttons region (indicates in-game)
            control_buttons_region = self._get_region('control_buttons')
            if control_buttons_region:
                x1, y1, x2, y2 = control_buttons_region
                if x2 > x1 and y2 > y1:
                    control_area = screenshot[y1:y2, x1:x2]
                    if control_area.size > 0:
                        # Check for button-like elements (edges/corners)
                        gray = _bgr_to_gray(control_area)
                        edges = _canny_edge_detection(gray, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        if edge_density > 0.05:  # Has enough edges (buttons visible)
                            detected['control_buttons'] = True
                        del edges
                        del gray
            
            # Check for quest panel region
            quests_region = self._get_region('quests')
            if quests_region:
                x1, y1, x2, y2 = quests_region
                if x2 > x1 and y2 > y1:
                    quest_area = screenshot[y1:y2, x1:x2]
                    if quest_area.size > 0:
                        detected['quest_panel'] = True
            
            # Note: Menu detection can be added here using OCR or region-based detection if needed
            # For now, we rely on region-based and text-based state detection
            
            return detected
            
        except Exception as e:
            logger.debug(f"Error detecting UI elements: {e}")
            return detected
    
    def _detect_state_from_text(self, screenshot: np.ndarray) -> Optional[str]:
        """
        Detect game state from OCR text patterns
        
        Returns:
            Detected state string or None
        """
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return None
            
            # Perform OCR on full resolution screenshot
            results = ocr_reader.readtext(screenshot, detail=1, paragraph=False)
            
            # Extract all text
            all_text = " ".join([text for (bbox, text, conf) in results]).lower()
            
            # Check each state pattern
            for state, patterns in self.state_text_patterns.items():
                for pattern in patterns:
                    if pattern in all_text:
                        logger.debug(f"State '{state}' detected from text pattern: '{pattern}'")
                        return state
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting state from text: {e}")
            return None
    
    def detect_auto_hunt_in_control_buttons(self, screenshot: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Detect auto_hunt_button image in the control_buttons region using template matching
        
        Args:
            screenshot: Full resolution screenshot
            
        Returns:
            Tuple of (is_detected, position) where position is (center_x, center_y) if found, None otherwise
        """
        try:
            # Get control_buttons region
            control_buttons_region = self._get_region('control_buttons')
            if not control_buttons_region:
                logger.debug("control_buttons region not configured")
                return False, None
            
            x1, y1, x2, y2 = control_buttons_region
            # Ensure coordinates are within screenshot bounds
            x1 = max(0, min(x1, screenshot.shape[1]))
            y1 = max(0, min(y1, screenshot.shape[0]))
            x2 = max(0, min(x2, screenshot.shape[1]))
            y2 = max(0, min(y2, screenshot.shape[0]))
            
            if x2 <= x1 or y2 <= y1:
                logger.debug("Invalid control_buttons region coordinates")
                return False, None
            
            # Convert region to (x, y, width, height) format for template matcher
            region_width = x2 - x1
            region_height = y2 - y1
            search_region = (x1, y1, region_width, region_height)
            
            # Try to find auto_hunt_button template in the control_buttons region
            # Use the actual template file name from assets/templates
            result = self.template_matcher.find_template(
                screenshot,
                "auto_hunt_button.png",
                confidence=0.7,  # Minimum confidence threshold
                region=search_region,
                multi_scale=True
            )
            
            if result:
                center_x, center_y, confidence = result
                logger.info(f"Detected auto_hunt_button image in control_buttons region at ({center_x}, {center_y}) (confidence: {confidence:.3f})")
                return True, (center_x, center_y)
            
            logger.debug("auto_hunt_button image not detected in control_buttons region")
            return False, None
                    
        except Exception as e:
            logger.debug(f"Error detecting AUTO HUNT button in control buttons: {e}")
            return False, None
    
    def _detect_state_from_regions(self, screenshot: np.ndarray) -> Optional[str]:
        """
        Detect game state based on visible regions
        
        Returns:
            Detected state string or None
        """
        try:
            # Check for key regions that indicate in-game state
            health_bar = self._get_region('health_bar')
            control_buttons = self._get_region('control_buttons')
            player_region = self._get_region('player')
            
            if health_bar and control_buttons:
                # Both health bar and control buttons visible = in-game
                x1, y1, x2, y2 = health_bar
                if x2 > x1 and y2 > y1:
                    health_area = screenshot[y1:y2, x1:x2]
                    if health_area.size > 0:
                        # Check if health bar has content
                        gray = _bgr_to_gray(health_area)
                        if np.std(gray) > 10:  # Has variation
                            logger.debug("In-game state detected: health bar and control buttons visible")
                            return 'in_game'
            
            # Check for player region (character visible)
            if player_region:
                x1, y1, x2, y2 = player_region
                if x2 > x1 and y2 > y1:
                    player_area = screenshot[y1:y2, x1:x2]
                    if player_area.size > 0:
                        # Check if player area has significant content (not empty)
                        gray = _bgr_to_gray(player_area)
                        avg_brightness = np.mean(gray)
                        std_dev = np.std(gray)
                        if std_dev > 15 and 50 < avg_brightness < 200:  # Has content
                            logger.debug("In-game state detected: player region visible")
                            return 'in_game'
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting state from regions: {e}")
            return None
    
    def _combine_state_evidence(self, ui_elements: Dict[str, bool], 
                               text_state: Optional[str],
                               region_state: Optional[str]) -> Optional[str]:
        """
        Combine multiple evidence sources to determine game state
        
        Returns:
            Final determined game state
        """
        # Priority 1: Region-based detection (most reliable)
        if region_state:
            return region_state
        
        # Priority 2: Text-based detection
        if text_state:
            return text_state
        
        # Priority 3: UI element detection
        if ui_elements.get('health_bar') and ui_elements.get('control_buttons'):
            return 'in_game'
        elif ui_elements.get('menu'):
            return 'main_menu'
        elif ui_elements.get('quest_panel'):
            return 'in_game_with_ui'
        
        # Default: unknown state
        return None
    
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
            # Process at full device resolution for accurate analysis
            # No downsampling - use full resolution screenshot
            screenshot_small = screenshot
            
            # Convert to different color spaces for analysis
            hsv_image = _bgr_to_hsv(screenshot_small)
            gray_image = _bgr_to_gray(screenshot_small)
            
            # Analyze dominant colors
            analysis['colors'] = self._analyze_colors(screenshot_small)
            
            # Detect UI elements (basic edge detection)
            edges = _canny_edge_detection(gray_image, 50, 150)
            contours = None  # Initialize contours variable
            try:
                if hasattr(cv2, 'findContours') and hasattr(cv2, 'RETR_EXTERNAL') and hasattr(cv2, 'CHAIN_APPROX_SIMPLE'):
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Count significant contours (potential UI elements)
                    if hasattr(cv2, 'contourArea'):
                        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
                        analysis['ui_elements'] = len(significant_contours)
                    else:
                        # Fallback: use edge density as proxy for UI elements
                        edge_density = np.sum(edges > 0) / edges.size
                        analysis['ui_elements'] = int(edge_density * 1000)  # Scale to approximate count
                else:
                    # Fallback: use edge density as proxy for UI elements
                    edge_density = np.sum(edges > 0) / edges.size
                    analysis['ui_elements'] = int(edge_density * 1000)  # Scale to approximate count
            except Exception as e:
                logger.debug(f"Error detecting contours: {e}")
                # Fallback: use edge density as proxy for UI elements
                edge_density = np.sum(edges > 0) / edges.size
                analysis['ui_elements'] = int(edge_density * 1000)  # Scale to approximate count
            
            # Basic game state detection based on colors and layout
            analysis['menu_state'] = self._detect_menu_state(screenshot_small, hsv_image)
            
            # Release temporary images explicitly
            if screenshot_small is not None and screenshot_small is not screenshot:
                del screenshot_small
            if hsv_image is not None:
                del hsv_image
            if gray_image is not None:
                del gray_image
            if edges is not None:
                del edges
            if contours is not None:
                del contours
            
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
            small_image = _resize_image(image, (100, 100))
            
            # Calculate mean colors
            mean_color = np.mean(small_image, axis=(0, 1))
            
            # Convert to different color spaces
            # Convert single color value to HSV
            color_array = np.uint8([[mean_color]])
            hsv_mean = _bgr_to_hsv(color_array)[0][0]
            
            return {
                'mean_bgr': mean_color.tolist(),
                'mean_hsv': hsv_mean.tolist(),
                'brightness': float(np.mean(_bgr_to_gray(small_image))),
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
            top_brightness = np.mean(_bgr_to_gray(top_region))
            bottom_brightness = np.mean(_bgr_to_gray(bottom_region))
            center_brightness = np.mean(_bgr_to_gray(center_region))
            
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