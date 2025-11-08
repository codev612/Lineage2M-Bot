"""
Game Detector Module - Detects Lineage 2M game state and status
Handles game detection, state analysis, and game-specific operations
"""

from typing import Tuple, Optional, Dict, List, Any
import re
import random
import cv2  # type: ignore
import numpy as np  # type: ignore
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
        from PIL import Image  # type: ignore
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
            from scipy import ndimage  # type: ignore
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
        
        # Track whether we've already detected the playing state to adjust future rules
        self._has_entered_playing = False
        
        # Track detected detailed game state (e.g., 'teleporting') without changing actual_game_state
        self._detected_detailed_state = None
        
        # Store tip window text positions for Rule 14
        self._tip_window_do_not_show_pos = None
        self._tip_window_confirm_pos = None
        
        # Store Accept button position detected in Rule 15
        self._source_accept_button_pos = None
        
        # Store Claim button position detected in Rule 17
        self._source_claim_button_pos = None
        
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
                    import torch  # type: ignore
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
                    pass
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
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
                    pass
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
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
                    pass
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            return False
    
    def _detect_check_death_penalty_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "check", "death", and "penalty" text in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            True if all three words ("check", "death", "penalty") are detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Collect all text from the region
                all_text = ""
                for (bbox, text, confidence) in results:
                    all_text += " " + text.lower().strip()
                
                all_text = all_text.lower().strip()
                
                # Check if all three words are present: "check", "death", and "penalty"
                words_to_find = ['check', 'death', 'penalty']
                found_words = []
                
                for word in words_to_find:
                    if word in all_text:
                        found_words.append(word)
                
                if len(found_words) == len(words_to_find):
                    logger.info(f"All required words detected in check_death_penalty region: {found_words}")
                    return True
                else:
                    return False
                
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            return False
    
    def _detect_claim_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "Claim" text in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            True if "Claim" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search for "Claim" text (case-insensitive)
                claim_keywords = ['claim']
                
                found = False
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Check if "claim" is in the text
                    for keyword in claim_keywords:
                        if keyword in text_lower:
                            logger.info(f"Detected 'Claim' text: '{text}' (confidence: {confidence:.3f})")
                            found = True
                            break
                    if found:
                        break
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting Claim text: {e}")
            return False
    
    def _detect_accept_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "Accept" text in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            True if "Accept" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search for "Accept" text (case-insensitive)
                accept_keywords = ['accept']
                
                found = False
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Check if "accept" is in the text
                    for keyword in accept_keywords:
                        if keyword in text_lower:
                            logger.info(f"Detected 'Accept' text: '{text}' (confidence: {confidence:.3f})")
                            found = True
                            break
                    if found:
                        break
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting Accept text: {e}")
            return False
    
    def _detect_do_not_show_again_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "Do not show again" text in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            True if "Do not show again" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search for "Do not show again" text (case-insensitive, can be partial match)
                keywords = ['do not show', 'do not show again', 'not show again']
                
                found = False
                all_text = ""
                for (bbox, text, confidence) in results:
                    all_text += " " + text.lower().strip()
                
                all_text = all_text.lower().strip()
                
                # Check if any keyword is in the text
                for keyword in keywords:
                    if keyword in all_text:
                        logger.info(f"Detected 'Do not show again' text: '{all_text}' (keyword: '{keyword}')")
                        found = True
                        break
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting Do not show again text: {e}")
            return False
    
    def _detect_confirm_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "confirm" text in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            True if "confirm" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search for "confirm" text (case-insensitive)
                confirm_keywords = ['confirm']
                
                found = False
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Check if "confirm" is in the text
                    for keyword in confirm_keywords:
                        if keyword in text_lower:
                            logger.info(f"Detected 'confirm' text: '{text}' (confidence: {confidence:.3f})")
                            found = True
                            break
                    if found:
                        break
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting confirm text: {e}")
            return False
    
    def _detect_teleport_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "?" character in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            True if "?" character is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search for "?" character
                found = False
                for (bbox, text, confidence) in results:
                    # Check if "?" is in the text
                    if '?' in text:
                        logger.info(f"Detected '?' character: '{text}' (confidence: {confidence:.3f})")
                        found = True
                        break
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting ? character: {e}")
            return False
    
    def _detect_resurrect_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "Resurrect" text in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            True if "Resurrect" text is detected, False otherwise
        """
        try:
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search for "Resurrect" text (case-insensitive)
                found = False
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Check if "resurrect" is in the text
                    if 'resurrect' in text_lower:
                        logger.info(f"Detected 'Resurrect' text: '{text}' (confidence: {confidence:.3f})")
                        found = True
                        break
                
                return found
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.debug(f"Error detecting Resurrect text: {e}")
            return False
    
    def _detect_do_not_show_again_text_position(self, region_image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect "Do not show again" text in a region using Tesseract OCR and return its position
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            Tuple of (x, y) center coordinates of the detected text, or None if not found
        """
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return None
            
            results = None
            try:
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search variations for "Do not show again"
                search_variations = [
                    "do not show again",
                    "do not show",
                    "not show again",
                    "donot show again",
                ]
                
                for bbox, text, confidence in results:
                    text_lower = text.lower().strip()
                    for variation in search_variations:
                        if variation in text_lower:
                            # Calculate center of bounding box
                            if len(bbox) >= 4:
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                                center_x = int(sum(x_coords) / len(x_coords))
                                center_y = int(sum(y_coords) / len(y_coords))
                                logger.debug(f"Detected 'Do not show again' text at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                                return (center_x, center_y)
                
                return None
            finally:
                if results is not None:
                    del results
                    
        except Exception as e:
            logger.error(f"Error detecting 'Do not show again' text position: {e}")
            return None
    
    def _detect_confirm_text_position(self, region_image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect "confirm" text in a region using Tesseract OCR and return its position
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            Tuple of (x, y) center coordinates of the detected text, or None if not found
        """
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return None
            
            results = None
            try:
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Search variations for "confirm"
                search_variations = [
                    "confirm",
                    "conﬁrm",  # sometimes OCR reads i as ﬁ
                ]
                
                for bbox, text, confidence in results:
                    text_lower = text.lower().strip()
                    for variation in search_variations:
                        if variation in text_lower:
                            # Calculate center of bounding box
                            if len(bbox) >= 4:
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                                center_x = int(sum(x_coords) / len(x_coords))
                                center_y = int(sum(y_coords) / len(y_coords))
                                logger.debug(f"Detected 'confirm' text at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                                return (center_x, center_y)
                
                return None
            finally:
                if results is not None:
                    del results
                    
        except Exception as e:
            logger.error(f"Error detecting 'confirm' text position: {e}")
            return None
    
    def _extract_number_from_region(self, region_image: np.ndarray, debug_name: str = None, save_debug_image: bool = False) -> Optional[float]:
        """
        Extract a number from a region image using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            debug_name: Optional name for debug logging (e.g., "price", "player_total")
            save_debug_image: If True, save the region image for debugging
            
        Returns:
            Extracted number as float, or None if no number found
        """
        try:
            # Save debug image if requested
            if save_debug_image and debug_name:
                try:
                    import datetime
                    from pathlib import Path
                    import cv2
                    debug_dir = Path('debug_screenshots')
                    debug_dir.mkdir(exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    debug_path = debug_dir / f"{debug_name}_region_{timestamp}.png"
                    cv2.imwrite(str(debug_path), region_image)
                    logger.info(f"Saved {debug_name} region image to: {debug_path}")
                except Exception as e:
                    logger.debug(f"Failed to save debug image: {e}")
            
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            import re
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                logger.warning(f"OCR reader not available for {debug_name or 'region'}")
                return None
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Combine all text and extract numbers
                all_text = ""
                all_text_details = []
                for (bbox, text, confidence) in results:
                    all_text += " " + text.strip()
                    all_text_details.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
                
                # Log raw OCR results for debugging
                if debug_name:
                    logger.info(f"[DEBUG {debug_name}] Raw OCR text: '{all_text}'")
                    logger.info(f"[DEBUG {debug_name}] OCR details: {all_text_details}")
                
                # Extract numbers (pattern matches digits with optional commas and decimal point)
                number_pattern = r'[\d,]+\.?\d*'
                matches = re.findall(number_pattern, all_text)
                
                if matches:
                    # Get the largest number (likely the main value)
                    numbers = []
                    for match in matches:
                        try:
                            # Remove commas and convert to float
                            num_str = match.replace(',', '').replace(' ', '').strip()
                            if num_str:
                                num = float(num_str)
                                numbers.append(num)
                        except ValueError:
                            continue
                    
                    if numbers:
                        # Return the largest number found
                        extracted_number = max(numbers)
                        if debug_name:
                            logger.info(f"[DEBUG {debug_name}] Extracted number: {extracted_number} from text: '{all_text}'")
                            logger.info(f"[DEBUG {debug_name}] All numbers found: {numbers}, using max: {extracted_number}")
                        else:
                            logger.info(f"Extracted number: {extracted_number} from text: '{all_text}'")
                        return extracted_number
                
                if debug_name:
                    logger.warning(f"[DEBUG {debug_name}] No number found in text: '{all_text}'")
                else:
                    logger.debug(f"No number found in text: '{all_text}'")
                return None
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.error(f"Error extracting number from region ({debug_name or 'unknown'}): {e}")
            return None
    
    def _extract_number_with_debug(self, region_image: np.ndarray, debug_name: str) -> dict:
        """
        Extract a number from a region image with detailed debug information
        
        Args:
            region_image: OpenCV image array (cropped region)
            debug_name: Name for debug logging (e.g., "price", "player_total")
            
        Returns:
            Dictionary with extraction results:
            {
                'number': float or None,
                'raw_text': str,
                'all_numbers': list,
                'ocr_details': list
            }
        """
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            import re
            import datetime
            from pathlib import Path
            import cv2
            
            # Save debug image
            try:
                debug_dir = Path('debug_screenshots')
                debug_dir.mkdir(exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                debug_path = debug_dir / f"{debug_name}_region_{timestamp}.png"
                cv2.imwrite(str(debug_path), region_image)
                logger.info(f"Saved {debug_name} region image to: {debug_path}")
            except Exception as e:
                logger.debug(f"Failed to save debug image: {e}")
            
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return {
                    'number': None,
                    'raw_text': '',
                    'all_numbers': [],
                    'ocr_details': [],
                    'error': 'OCR reader not available'
                }
            
            results = None
            try:
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                all_text = ""
                all_text_details = []
                for (bbox, text, confidence) in results:
                    all_text += " " + text.strip()
                    all_text_details.append({
                        'text': text.strip(),
                        'confidence': float(confidence),
                        'bbox': [int(b) for b in bbox]
                    })
                
                number_pattern = r'[\d,]+\.?\d*'
                matches = re.findall(number_pattern, all_text)
                
                numbers = []
                for match in matches:
                    try:
                        num_str = match.replace(',', '').replace(' ', '').strip()
                        if num_str:
                            num = float(num_str)
                            numbers.append(num)
                    except ValueError:
                        continue
                
                extracted_number = max(numbers) if numbers else None
                
                return {
                    'number': extracted_number,
                    'raw_text': all_text.strip(),
                    'all_numbers': numbers,
                    'ocr_details': all_text_details,
                    'error': None
                }
            finally:
                if results is not None:
                    del results
                    
        except Exception as e:
            logger.error(f"Error in _extract_number_with_debug ({debug_name}): {e}")
            return {
                'number': None,
                'raw_text': '',
                'all_numbers': [],
                'ocr_details': [],
                'error': str(e)
            }
    
    def _detect_accept_quest_text(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect "Accept Quest" text in screenshot using Tesseract OCR
        
        Args:
            screenshot: OpenCV image array (full screenshot or region)
            
        Returns:
            Tuple of (x, y) center coordinates of the detected text, or None if not found
        """
        try:
            # Save debug screenshot of the region being analyzed
            debug_dir = Path('debug_screenshots')
            debug_dir.mkdir(exist_ok=True)
            import time
            timestamp = int(time.time() * 1000)
            debug_path = debug_dir / f"accept_quest_text_search_{timestamp}.png"
            self._save_image_rgb(screenshot, debug_path)
            logger.debug(f"Saved debug screenshot for Accept Quest text detection to {debug_path}")
            
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                logger.warning("Tesseract OCR reader not available for Accept Quest text detection")
                return None
            
            results = None
            try:
                # Perform OCR on the screenshot
                results = ocr_reader.readtext(screenshot, detail=1, paragraph=False)
                
                # Debug: Log all OCR results
                logger.debug(f"OCR detected {len(results)} text elements for Accept Quest detection:")
                all_texts = []
                all_text_bboxes = []
                for idx, (bbox, text, confidence) in enumerate(results):
                    text_lower = text.lower().strip()
                    if text_lower:  # Only add non-empty text
                        all_texts.append(text_lower)
                        all_text_bboxes.append(bbox)
                        # Calculate center of bounding box for logging
                        if len(bbox) >= 4:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            center_x = int(sum(x_coords) / len(x_coords))
                            center_y = int(sum(y_coords) / len(y_coords))
                            logger.debug(f"  [{idx}] Text: '{text}' (lower: '{text_lower}') at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                
                # Combine all text into a single string for searching
                combined_text = " ".join(all_texts)
                logger.debug(f"All detected texts (combined): '{combined_text}'")
                
                # Look for "Accept Quest" text - try multiple search strategies
                search_variations = [
                    "accept quest",
                    "acceptquest",
                    "accept  quest",  # with double space
                ]
                
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
                                logger.info(f"Detected 'Accept Quest' text in single result at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                                
                                # Save debug screenshot with annotation
                                debug_annotated = screenshot.copy()
                                bbox_int = np.array(bbox, dtype=np.int32)
                                cv2.polylines(debug_annotated, [bbox_int], True, (0, 255, 0), 2)
                                cv2.circle(debug_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                                cv2.putText(debug_annotated, f"Accept Quest ({confidence:.2f})", 
                                          (center_x - 50, center_y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                annotated_path = debug_dir / f"accept_quest_text_detected_{timestamp}.png"
                                self._save_image_rgb(debug_annotated, annotated_path)
                                logger.debug(f"Saved annotated debug screenshot to {annotated_path}")
                                
                                return (center_x, center_y)
                
                # Strategy 2: Check if search text appears in combined text
                # If found, find the center of all bounding boxes that contain "accept" or "quest"
                for variation in search_variations:
                    if variation in combined_text:
                        logger.info(f"Found '{variation}' in combined text - locating position...")
                        # Find all bboxes that contain "accept" or "quest"
                        matching_bboxes = []
                        matching_texts = []
                        for idx, (bbox, text, confidence) in enumerate(results):
                            text_lower = text.lower().strip()
                            if "accept" in text_lower or "quest" in text_lower:
                                matching_bboxes.append(bbox)
                                matching_texts.append(text_lower)
                        
                        if matching_bboxes:
                            # Calculate combined center of all matching bboxes
                            all_x_coords = []
                            all_y_coords = []
                            for bbox in matching_bboxes:
                                if len(bbox) >= 4:
                                    x_coords = [point[0] for point in bbox]
                                    y_coords = [point[1] for point in bbox]
                                    all_x_coords.extend(x_coords)
                                    all_y_coords.extend(y_coords)
                            
                            if all_x_coords and all_y_coords:
                                center_x = int(sum(all_x_coords) / len(all_x_coords))
                                center_y = int(sum(all_y_coords) / len(all_y_coords))
                                logger.info(f"Detected 'Accept Quest' text in combined results at ({center_x}, {center_y})")
                                logger.debug(f"Matching texts: {', '.join(matching_texts)}")
                                
                                # Save debug screenshot with annotation
                                debug_annotated = screenshot.copy()
                                # Draw all matching bboxes
                                for bbox in matching_bboxes:
                                    bbox_int = np.array(bbox, dtype=np.int32)
                                    cv2.polylines(debug_annotated, [bbox_int], True, (0, 255, 0), 2)
                                # Draw center point
                                cv2.circle(debug_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                                cv2.putText(debug_annotated, f"Accept Quest (combined)", 
                                          (center_x - 50, center_y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                annotated_path = debug_dir / f"accept_quest_text_detected_{timestamp}.png"
                                self._save_image_rgb(debug_annotated, annotated_path)
                                logger.debug(f"Saved annotated debug screenshot to {annotated_path}")
                                
                                return (center_x, center_y)
                
                # Strategy 3: Check if "accept" and "quest" appear separately but nearby
                accept_bbox = None
                quest_bbox = None
                accept_text = None
                quest_text = None
                
                for idx, (bbox, text, confidence) in enumerate(results):
                    text_lower = text.lower().strip()
                    if text_lower == "accept" or text_lower.startswith("accept"):
                        accept_bbox = bbox
                        accept_text = text_lower
                    elif text_lower == "quest" or text_lower.startswith("quest"):
                        quest_bbox = bbox
                        quest_text = text_lower
                
                if accept_bbox and quest_bbox:
                    # Check if they are nearby (within reasonable distance)
                    if len(accept_bbox) >= 4 and len(quest_bbox) >= 4:
                        acc_x_coords = [point[0] for point in accept_bbox]
                        acc_y_coords = [point[1] for point in accept_bbox]
                        acc_center_x = int(sum(acc_x_coords) / len(acc_x_coords))
                        acc_center_y = int(sum(acc_y_coords) / len(acc_y_coords))
                        
                        que_x_coords = [point[0] for point in quest_bbox]
                        que_y_coords = [point[1] for point in quest_bbox]
                        que_center_x = int(sum(que_x_coords) / len(que_x_coords))
                        que_center_y = int(sum(que_y_coords) / len(que_y_coords))
                        
                        # Calculate distance between centers
                        distance = ((acc_center_x - que_center_x) ** 2 + (acc_center_y - que_center_y) ** 2) ** 0.5
                        # Check if they're on similar Y coordinates (same line) and within reasonable X distance
                        y_diff = abs(acc_center_y - que_center_y)
                        max_y_diff = screenshot.shape[0] * 0.1  # 10% of screenshot height
                        max_distance = screenshot.shape[1] * 0.5  # 50% of screenshot width
                        
                        if y_diff < max_y_diff and distance < max_distance:
                            # Calculate center between the two
                            center_x = (acc_center_x + que_center_x) // 2
                            center_y = (acc_center_y + que_center_y) // 2
                            logger.info(f"Detected 'Accept' and 'Quest' as separate nearby texts at ({center_x}, {center_y})")
                            logger.debug(f"'Accept' at ({acc_center_x}, {acc_center_y}), 'Quest' at ({que_center_x}, {que_center_y}), distance: {distance:.1f}")
                            
                            # Save debug screenshot with annotation
                            debug_annotated = screenshot.copy()
                            # Draw both bboxes
                            acc_bbox_int = np.array(accept_bbox, dtype=np.int32)
                            que_bbox_int = np.array(quest_bbox, dtype=np.int32)
                            cv2.polylines(debug_annotated, [acc_bbox_int], True, (0, 255, 0), 2)
                            cv2.polylines(debug_annotated, [que_bbox_int], True, (0, 255, 0), 2)
                            # Draw center point
                            cv2.circle(debug_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                            cv2.putText(debug_annotated, f"Accept Quest (separate)", 
                                      (center_x - 50, center_y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            annotated_path = debug_dir / f"accept_quest_text_detected_{timestamp}.png"
                            self._save_image_rgb(debug_annotated, annotated_path)
                            logger.debug(f"Saved annotated debug screenshot to {annotated_path}")
                            
                            return (center_x, center_y)
                
                logger.debug(f"'Accept Quest' text not found in OCR results. Searched for: {search_variations}")
                logger.debug(f"Combined text was: '{combined_text}'")
                return None
                
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.error(f"Error detecting Accept Quest text: {e}", exc_info=True)
            return None
    
    def _detect_general_merchant_text(self, region_image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect "General" text in a region using Tesseract OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            Tuple of (x, y) center coordinates of the detected text, or None if not found
        """
        try:
            # Save debug screenshot of the region being analyzed
            debug_dir = Path('debug_screenshots')
            debug_dir.mkdir(exist_ok=True)
            import time
            timestamp = int(time.time() * 1000)
            debug_path = debug_dir / f"general_merchant_region_{timestamp}.png"
            self._save_image_rgb(region_image, debug_path)
            logger.debug(f"Saved debug screenshot of merchant_list_region to {debug_path}")
            
            # Use Tesseract OCR reader
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                logger.warning("Tesseract OCR reader not available for General detection")
                return None
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Debug: Log all OCR results
                logger.debug(f"OCR detected {len(results)} text elements in merchant_list_region:")
                all_texts = []
                all_text_bboxes = []
                for idx, (bbox, text, confidence) in enumerate(results):
                    text_lower = text.lower().strip()
                    if text_lower:  # Only add non-empty text
                        all_texts.append(text_lower)
                        all_text_bboxes.append(bbox)
                        # Calculate center of bounding box for logging
                        if len(bbox) >= 4:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            center_x = int(sum(x_coords) / len(x_coords))
                            center_y = int(sum(y_coords) / len(y_coords))
                            logger.debug(f"  [{idx}] Text: '{text}' (lower: '{text_lower}') at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                
                # Combine all text into a single string for searching
                combined_text = " ".join(all_texts)
                logger.debug(f"All detected texts (combined): '{combined_text}'")
                
                # Look for "General" text - try multiple search strategies
                search_text = "general"
                search_variations = [
                    "general",
                    "genera",  # in case of partial detection
                ]
                
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
                                logger.info(f"Detected 'General' text in single result at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                                
                                # Save debug screenshot with annotation
                                debug_annotated = region_image.copy()
                                bbox_int = np.array(bbox, dtype=np.int32)
                                cv2.polylines(debug_annotated, [bbox_int], True, (0, 255, 0), 2)
                                cv2.circle(debug_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                                cv2.putText(debug_annotated, f"General ({confidence:.2f})", 
                                          (center_x - 50, center_y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                annotated_path = debug_dir / f"general_merchant_detected_{timestamp}.png"
                                self._save_image_rgb(debug_annotated, annotated_path)
                                logger.debug(f"Saved annotated debug screenshot to {annotated_path}")
                                
                                return (center_x, center_y)
                
                # Strategy 2: Check if search text appears in combined text
                # If found, find the center of all bounding boxes that contain "general"
                for variation in search_variations:
                    if variation in combined_text:
                        logger.info(f"Found '{variation}' in combined text - locating position...")
                        # Find all bboxes that contain "general"
                        matching_bboxes = []
                        matching_texts = []
                        for idx, (bbox, text, confidence) in enumerate(results):
                            text_lower = text.lower().strip()
                            if "general" in text_lower:
                                matching_bboxes.append(bbox)
                                matching_texts.append(text_lower)
                        
                        if matching_bboxes:
                            # Calculate combined center of all matching bboxes
                            all_x_coords = []
                            all_y_coords = []
                            for bbox in matching_bboxes:
                                if len(bbox) >= 4:
                                    x_coords = [point[0] for point in bbox]
                                    y_coords = [point[1] for point in bbox]
                                    all_x_coords.extend(x_coords)
                                    all_y_coords.extend(y_coords)
                            
                            if all_x_coords and all_y_coords:
                                center_x = int(sum(all_x_coords) / len(all_x_coords))
                                center_y = int(sum(all_y_coords) / len(all_y_coords))
                                logger.info(f"Detected 'General' text in combined results at ({center_x}, {center_y})")
                                logger.debug(f"Matching texts: {', '.join(matching_texts)}")
                                
                                # Save debug screenshot with annotation
                                debug_annotated = region_image.copy()
                                # Draw all matching bboxes
                                for bbox in matching_bboxes:
                                    bbox_int = np.array(bbox, dtype=np.int32)
                                    cv2.polylines(debug_annotated, [bbox_int], True, (0, 255, 0), 2)
                                # Draw center point
                                cv2.circle(debug_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                                cv2.putText(debug_annotated, f"General (combined)", 
                                          (center_x - 50, center_y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                annotated_path = debug_dir / f"general_merchant_detected_{timestamp}.png"
                                self._save_image_rgb(debug_annotated, annotated_path)
                                logger.debug(f"Saved annotated debug screenshot to {annotated_path}")
                                
                                return (center_x, center_y)
                
                # Strategy 3: Check if "general" appears as exact match or starts with "general"
                for idx, (bbox, text, confidence) in enumerate(results):
                    text_lower = text.lower().strip()
                    if text_lower == "general" or text_lower.startswith("general"):
                        # Calculate center of bounding box
                        if len(bbox) >= 4:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            center_x = int(sum(x_coords) / len(x_coords))
                            center_y = int(sum(y_coords) / len(y_coords))
                            logger.info(f"Detected 'General' text (exact/prefix match) at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                            
                            # Save debug screenshot with annotation
                            debug_annotated = region_image.copy()
                            bbox_int = np.array(bbox, dtype=np.int32)
                            cv2.polylines(debug_annotated, [bbox_int], True, (0, 255, 0), 2)
                            cv2.circle(debug_annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                            cv2.putText(debug_annotated, f"General (exact)", 
                                      (center_x - 50, center_y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            annotated_path = debug_dir / f"general_merchant_detected_{timestamp}.png"
                            self._save_image_rgb(debug_annotated, annotated_path)
                            logger.debug(f"Saved annotated debug screenshot to {annotated_path}")
                            
                            return (center_x, center_y)
                
                logger.debug(f"'General' text not found in OCR results. Searched for: {search_variations}")
                logger.debug(f"Combined text was: '{combined_text}'")
                return None
                
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            logger.error(f"Error detecting General Merchant text: {e}", exc_info=True)
            return None
    
    def detect_player_parameters(self, screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect player parameters from configured regions:
        - blood_bottle: Extract number from blood_bottle_number_region
        - bag: Extract weight from bag region
        
        Args:
            screenshot: OpenCV image array (full screenshot)
            
        Returns:
            Dictionary with detected parameters:
            - 'blood_bottle': str or None (extracted number)
            - 'bag_weight': str or None (extracted weight)
            Or None if no regions configured
        """
        try:
            # Load regions if not loaded
            if not self.regions and self.adb.device_id:
                self._load_regions(self.adb.device_id)
            
            parameters = {}
            
            # Extract blood bottle number
            blood_bottle_region = self._get_region('blood_bottle_number_region')
            if blood_bottle_region:
                x1, y1, x2, y2 = blood_bottle_region
                # Ensure coordinates are within screenshot bounds
                x1 = max(0, min(x1, screenshot.shape[1]))
                y1 = max(0, min(y1, screenshot.shape[0]))
                x2 = max(0, min(x2, screenshot.shape[1]))
                y2 = max(0, min(y2, screenshot.shape[0]))
                
                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    blood_bottle_text = self._extract_text_from_region(region_image)
                    if blood_bottle_text:
                        # Extract only digits from the text
                        filtered_numbers = re.sub(r'[^\d]', '', blood_bottle_text)
                        if filtered_numbers:
                            parameters['blood_bottle'] = filtered_numbers
                        else:
                            parameters['blood_bottle'] = '0'  # Default to 0 if no numbers found
                    else:
                        parameters['blood_bottle'] = '0'  # Default to 0 if no text extracted
            
            # Extract bag weight
            bag_region = self._get_region('bag_weight_region')
            if bag_region:
                x1, y1, x2, y2 = bag_region
                # Ensure coordinates are within screenshot bounds
                x1 = max(0, min(x1, screenshot.shape[1]))
                y1 = max(0, min(y1, screenshot.shape[0]))
                x2 = max(0, min(x2, screenshot.shape[1]))
                y2 = max(0, min(y2, screenshot.shape[0]))
                
                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    bag_text = self._extract_text_from_region(region_image)
                    if bag_text:
                        # Extract only digits and forward slash (for patterns like "123/456")
                        filtered_text = re.sub(r'[^\d/]', '', bag_text)
                        if filtered_text:
                            parameters['bag_weight'] = filtered_text
                        else:
                            parameters['bag_weight'] = '0'  # Default to 0 if no numbers found
                    else:
                        parameters['bag_weight'] = '0'  # Default to 0 if no text extracted
            return parameters if parameters else None
            
        except Exception as e:
            logger.error(f"Error detecting player parameters: {e}", exc_info=True)
            return None
    
    def _extract_text_from_region(self, region_image: np.ndarray) -> Optional[str]:
        """
        Extract text from a region image using OCR
        
        Args:
            region_image: OpenCV image array (cropped region)
            
        Returns:
            Extracted text string or None
        """
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return None
            
            results = None
            try:
                # Perform OCR on the region image
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)
                
                # Collect all text from the region
                texts = []
                for (bbox, text, confidence) in results:
                    if text.strip():  # Only add non-empty text
                        texts.append(text.strip())
                
                if texts:
                    extracted_text = " ".join(texts)
                    return extracted_text
                else:
                    return None
                
            finally:
                # Always release OCR results immediately
                if results is not None:
                    del results
            
        except Exception as e:
            return None
    
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
            return False
    
    def detect_and_tap_unknown_state_buttons(self, screenshot: np.ndarray) -> bool:
        """Deprecated: unknown-state workflows have been removed."""
        logger.debug("Unknown state handling is disabled; skipping button detection.")
        return False
    
    def _detect_and_swipe_unlock_screen(self, screenshot: np.ndarray) -> bool:
        """
        Detect unlock screen region and perform a random swipe within it
        
        Args:
            screenshot: OpenCV image array (full screenshot)
            
        Returns:
            True if unlock screen region was detected and swiped, False otherwise
        """
        try:
            logger.info("Checking for unlock screen region...")
            
            # Find the unlock screen region template
            result = self.template_matcher.find_template(
                screenshot,
                "unlock_screen_region.png",
                multi_scale=True,
                confidence=0.7
            )
            
            if not result:
                logger.debug("Unlock screen region not detected")
                return False
            
            center_x, center_y, confidence = result
            logger.info(f"Unlock screen region detected at ({center_x}, {center_y}) with confidence {confidence:.3f}")
            
            # Load template to get its dimensions
            template = self.template_matcher.load_template("unlock_screen_region.png")
            if template is None:
                logger.warning("Could not load unlock_screen_region.png template to get dimensions")
                return False
            
            # Get template dimensions
            template_height, template_width = template.shape[:2]
            
            # Calculate region bounds (template center is at center_x, center_y)
            # Region extends from center by half the template size
            region_x1 = max(0, center_x - template_width // 2)
            region_y1 = max(0, center_y - template_height // 2)
            region_x2 = min(screenshot.shape[1], center_x + template_width // 2)
            region_y2 = min(screenshot.shape[0], center_y + template_height // 2)
            
            # Ensure we have a valid region with some size
            if region_x2 <= region_x1 or region_y2 <= region_y1:
                logger.warning(f"Invalid unlock screen region bounds: ({region_x1}, {region_y1}, {region_x2}, {region_y2})")
                return False
            
            # Generate random start and end points within the region
            # Add some padding to avoid edges (10% padding)
            padding_x = int((region_x2 - region_x1) * 0.1)
            padding_y = int((region_y2 - region_y1) * 0.1)
            
            swipe_x1 = random.randint(region_x1 + padding_x, region_x2 - padding_x)
            swipe_y1 = random.randint(region_y1 + padding_y, region_y2 - padding_y)
            swipe_x2 = random.randint(region_x1 + padding_x, region_x2 - padding_x)
            swipe_y2 = random.randint(region_y1 + padding_y, region_y2 - padding_y)
            
            # Ensure start and end points are different
            if swipe_x1 == swipe_x2 and swipe_y1 == swipe_y2:
                # If same point, add small offset
                offset = min(50, (region_x2 - region_x1) // 4)
                swipe_x2 = min(region_x2 - padding_x, swipe_x2 + random.randint(-offset, offset))
                swipe_y2 = min(region_y2 - padding_y, swipe_y2 + random.randint(-offset, offset))
            
            logger.info(f"Swiping in unlock screen region: from ({swipe_x1}, {swipe_y1}) to ({swipe_x2}, {swipe_y2})")
            
            # Perform the swipe using ADB
            if self.adb and hasattr(self.adb, 'execute_adb_command'):
                # Swipe duration in milliseconds (300ms default)
                duration = 300
                success, output = self.adb.execute_adb_command([
                    'shell', 'input', 'swipe',
                    str(swipe_x1), str(swipe_y1),
                    str(swipe_x2), str(swipe_y2),
                    str(duration)
                ])
                
                if success:
                    logger.info(f"Successfully swiped in unlock screen region from ({swipe_x1}, {swipe_y1}) to ({swipe_x2}, {swipe_y2})")
                    return True
                else:
                    logger.warning(f"Failed to swipe in unlock screen region: {output}")
                    return False
            else:
                logger.warning("ADB manager not available or execute_adb_command method not found")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting/swiping unlock screen region: {e}", exc_info=True)
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
        
        Basic main loop rules (checked in priority order):
        5. If screenshot includes unlock_screen_region.png → "screen_lock" (swipe screen) - checked first
        6. If "Skip" text in skip_button_region → tap button, set detailed_game_state to "unknown" (actual_game_state remains "playing")
        7. If "Claim" text in claim_reward_button_region → tap button and treat as playing with detailed_game_state "claim_reward"
        8. If "Accept" text in reward_accept_button_region → tap button and treat as playing with detailed_game_state "accepting_reward"
        9. If close_cross.png detected → "opened_dialogue" (tap button)
        10. If "Do not show again" text in do_not_show_again_region AND "confirm" text in tip_joining_confirm_region → "confirming_tips" (tap both)
        11. If do_not_show.png AND confirm_button_2.png detected in tip_window_region → store tap positions, return "confirming_tips_window"/"confirming_tips_window_partial"
        12. If short_button.png AND "?" text in confirm_description_region → tap confirm button, set detailed_game_state to "auto_questing" if tap succeeds (actual_game_state remains "playing")
        13. If game is in "playing" state AND "Resurrect" text in resurrect_button_region → tap resurrect_button_region, set detailed_game_state to "purchasing" if tap succeeds (actual_game_state remains "playing")
        1. If screenshot includes "tap" text AND select_server_button.png → "select_server"
        2. If screenshot includes "Character" text AND enter_button.png → "select_character"
        3. If screenshot includes short_button.png → "playing"
        4. Otherwise → "unknown"
        
        Args:
            screenshot: OpenCV image array
            debug: If True, save screenshot and detailed detection results
            
        Returns:
            Game state string: 
            - 'select_server', 'select_character', 'screen_lock', 'claim_reward', 'accepting_reward', 'opened_dialogue', 'confirming_tips', 'unknown'
            - 'playing' (in-game state)
            
        Note:
            Rule 10 sets detailed_game_state to 'teleporting' but returns 'playing' for actual_game_state.
            Use get_detected_detailed_state() to retrieve the detected detailed state.
        """
        try:
            logger.info("Starting game state detection from screenshot...")
            
            # Clear detected detailed state from previous detection
            self._detected_detailed_state = None
            
            # Collect annotation data for debug images
            annotations = {'templates': [], 'text': []}
            
            # Save screenshot for debugging if requested
            if debug:
                self._save_debug_screenshot(screenshot, "detection_start")
            
            tap_detected = False
            select_server_button_detected = False
            character_text_detected = False
            enter_button_detected = False
            unlock_screen_detected = False
            claim_reward_detected = False
            accepting_reward_detected = False
            close_cross_detected = False
            do_not_show_detected = False
            confirm_tip_detected = False
            teleport_short_button_detected = False
            teleport_text_detected = False
            resurrect_text_detected = False
            
            # Template matching confidences
            do_not_show_template_confidence = 0.7
            confirm_button_template_confidence = 0.7
            accept_confidence = 0.7
            claim_confidence = 0.7

            # Always check for screen lock first (even if already in playing state)
            logger.info("Rule 5: Checking for unlock_screen_region.png...")
            unlock_result = self.template_matcher.find_template(
                screenshot,
                "unlock_screen_region.png",
                multi_scale=True,
                confidence=0.7
            )
            unlock_screen_detected = unlock_result is not None
            if unlock_screen_detected and unlock_result:
                x, y, confidence = unlock_result
                logger.info(f"  - unlock_screen_region.png detected at ({x}, {y}) with confidence {confidence:.3f}")
                if debug:
                    annotations['templates'].append({
                        'name': 'unlock_screen_region',
                        'position': [int(x), int(y)],
                        'confidence': float(confidence),
                        'size': [0, 0]
                    })
                detected_state = 'screen_lock'
                logger.info(f"Game state detected: {detected_state} (unlock_screen_region.png found)")
                if debug:
                    self._save_debug_screenshot(screenshot, f"detected_{detected_state}", annotations)
                    self._save_template_images(['unlock_screen_region.png'])
                return detected_state
            else:
                logger.info(f"  - unlock_screen_region.png detected: {unlock_screen_detected}")
            
            # Rule 6: Check for "Skip" text in skip_button_region
            logger.info("Rule 6: Checking for 'Skip' text in skip_button_region...")
            skip_button_region = self._get_region('skip_button_region')
            if skip_button_region:
                x1, y1, x2, y2 = skip_button_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))
                
                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    skip_button_detected = self._detect_skip_text(region_image)
                    if skip_button_detected:
                        logger.info("  - 'Skip' text detected in skip_button_region")
                        if debug:
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            annotations['text'].append({
                                'text': 'Skip',
                                'position': [center_x, center_y],
                                'confidence': 1.0
                            })
                    else:
                        logger.info("  - 'Skip' text not detected in skip_button_region")
                else:
                    logger.debug("Invalid skip_button_region coordinates")
            else:
                logger.debug("skip_button_region not configured")

            if skip_button_detected:
                logger.info("Rule 6: 'Skip' text found - attempting to tap skip button and reset detailed state")
                tap_success = False
                if skip_button_region and self.adb and hasattr(self.adb, 'tap'):
                    x1, y1, x2, y2 = skip_button_region
                    tap_x = (x1 + x2) // 2
                    tap_y = (y1 + y2) // 2
                    screen_width = screenshot.shape[1]
                    screen_height = screenshot.shape[0]
                    if 0 <= tap_x < screen_width and 0 <= tap_y < screen_height:
                        logger.info(f"[RULE 6] Tapping skip button at ({tap_x}, {tap_y})")
                        tap_success = self.adb.tap(int(tap_x), int(tap_y))
                        if tap_success:
                            logger.info("[RULE 6] Successfully tapped skip button")
                        else:
                            logger.warning("[RULE 6] Failed to tap skip button")
                    else:
                        logger.warning(f"[RULE 6] Invalid skip_button_region coordinates: ({tap_x}, {tap_y})")
                else:
                    if not skip_button_region:
                        logger.warning("[RULE 6] skip_button_region not configured")
                    if not (self.adb and hasattr(self.adb, 'tap')):
                        logger.warning("[RULE 6] ADB manager not available or tap method not found")

                self._detected_detailed_state = 'unknown'
                logger.info("Rule 6: Setting detailed_game_state to 'unknown'")

                if not self._has_entered_playing:
                    logger.debug("Marking playing state as reached via Rule 6")
                    self._has_entered_playing = True

                if debug:
                    self._save_debug_screenshot(screenshot, "detected_skip_button", annotations)

                return 'playing'

            # Rule 7: Check for "Claim" text in claim_reward_button_region
            logger.info("Rule 7: Checking for 'Claim' text in claim_reward_button_region...")
            
            # Check for "Claim" text in claim_reward_button_region using OCR
            claim_reward_region = self._get_region('claim_reward_button_region')
            if claim_reward_region:
                x1, y1, x2, y2 = claim_reward_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))
                
                if x2 > x1 and y2 > y1:
                    # Extract region from screenshot
                    region_image = screenshot[y1:y2, x1:x2]
                    
                    # Use OCR to detect "Claim" text in the region
                    claim_reward_detected = self._detect_claim_text(region_image)
                    if claim_reward_detected:
                        logger.info(f"  - 'Claim' text detected in claim_reward_button_region")
                        if debug:
                            # Calculate center position for annotation
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            annotations['text'].append({
                                'text': 'Claim',
                                'position': [center_x, center_y],
                                'confidence': 1.0
                            })
                    else:
                        logger.info(f"  - 'Claim' text not detected in claim_reward_button_region")
                else:
                    logger.debug("Invalid claim_reward_button_region coordinates")
            else:
                logger.debug("claim_reward_button_region not configured")
            
            if claim_reward_detected:
                logger.info("Rule 7: 'Claim' text found - treating as playing with detailed state 'claim_reward'")
                self._detected_detailed_state = 'claim_reward'
                if not self._has_entered_playing:
                    logger.debug("Marking playing state as reached via Rule 7")
                    self._has_entered_playing = True
                if debug:
                    self._save_debug_screenshot(screenshot, "detected_claim_reward", annotations)
                return 'playing'
            
            # Rule 8: Check for "Accept" text in reward_accept_button_region
            logger.info("Rule 8: Checking for 'Accept' text in reward_accept_button_region...")
            accepting_reward_detected = False
            
            # Check for "Accept" text in reward_accept_button_region using OCR
            reward_accept_region = self._get_region('reward_accept_button_region')
            if reward_accept_region:
                x1, y1, x2, y2 = reward_accept_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))
                
                if x2 > x1 and y2 > y1:
                    # Extract region from screenshot
                    region_image = screenshot[y1:y2, x1:x2]
                    
                    # Use OCR to detect "Accept" text in the region
                    accepting_reward_detected = self._detect_accept_text(region_image)
                    if accepting_reward_detected:
                        logger.info(f"  - 'Accept' text detected in reward_accept_button_region")
                        if debug:
                            # Calculate center position for annotation
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            annotations['text'].append({
                                'text': 'Accept',
                                'position': [center_x, center_y],
                                'confidence': 1.0
                            })
                    else:
                        logger.info(f"  - 'Accept' text not detected in reward_accept_button_region")
                else:
                    logger.debug("Invalid reward_accept_button_region coordinates")
            else:
                logger.debug("reward_accept_button_region not configured")
            
            if accepting_reward_detected:
                logger.info("Rule 8: 'Accept' text found - treating as playing with detailed state 'accepting_reward'")
                self._detected_detailed_state = 'accepting_reward'
                if not self._has_entered_playing:
                    logger.debug("Marking playing state as reached via Rule 8")
                    self._has_entered_playing = True
                if debug:
                    self._save_debug_screenshot(screenshot, "detected_accepting_reward", annotations)
                return 'playing'
            
            # Rule 9: Check for close_cross.png
            logger.info("Rule 9: Checking for close_cross.png...")
            close_cross_detected = False
            close_cross_position = None
            close_cross_confidence = 0.0
            
            close_cross_result = self.template_matcher.find_template(
                screenshot,
                "close_cross.png",
                multi_scale=True,
                confidence=0.9
            )
            
            if close_cross_result:
                close_cross_detected = True
                close_cross_position = (int(close_cross_result[0]), int(close_cross_result[1]))
                close_cross_confidence = float(close_cross_result[2])
                logger.info(f"  - close_cross.png detected at {close_cross_position} with confidence {close_cross_confidence:.3f}")
                if debug:
                    annotations['templates'].append({
                        'name': 'close_cross',
                        'position': list(close_cross_position),
                        'confidence': close_cross_confidence,
                        'size': [0, 0]
                    })
            else:
                logger.info(f"  - close_cross.png not detected")
            
            if close_cross_detected:
                detected_state = 'opened_dialogue'
                logger.info(f"Game state detected: {detected_state} (close_cross.png found)")
                if debug:
                    self._save_debug_screenshot(screenshot, f"detected_{detected_state}", annotations)
                    self._save_template_images(['close_cross.png'])
                return detected_state
            
            # Rule 10: Check for "Do not show again" text in do_not_show_again_region AND "confirm" text in tip_joining_confirm_region
            logger.info("Rule 10: Checking for 'Do not show again' text in do_not_show_again_region AND 'confirm' text in tip_joining_confirm_region...")
            
            # Check for "Do not show again" text in do_not_show_again_region
            do_not_show_region = self._get_region('do_not_show_again_region')
            if do_not_show_region:
                x1, y1, x2, y2 = do_not_show_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))
                
                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    do_not_show_detected = self._detect_do_not_show_again_text(region_image)
                    if do_not_show_detected:
                        logger.info(f"  - 'Do not show again' text detected in do_not_show_again_region")
                        if debug:
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            annotations['text'].append({
                                'text': 'Do not show again',
                                'position': [center_x, center_y],
                                'confidence': 1.0
                            })
                    else:
                        logger.info(f"  - 'Do not show again' text not detected in do_not_show_again_region")
                else:
                    logger.debug("Invalid do_not_show_again_region coordinates")
            else:
                logger.debug("do_not_show_again_region not configured")
            
            # Check for "confirm" text in tip_joining_confirm_region
            tip_confirm_region = self._get_region('tip_joining_confirm_region')
            if tip_confirm_region:
                x1, y1, x2, y2 = tip_confirm_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))
                
                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    confirm_tip_detected = self._detect_confirm_text(region_image)
                    if confirm_tip_detected:
                        logger.info(f"  - 'confirm' text detected in tip_joining_confirm_region")
                        if debug:
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            annotations['text'].append({
                                'text': 'confirm',
                                'position': [center_x, center_y],
                                'confidence': 1.0
                            })
                    else:
                        logger.info(f"  - 'confirm' text not detected in tip_joining_confirm_region")
                else:
                    logger.debug("Invalid tip_joining_confirm_region coordinates")
            else:
                logger.debug("tip_joining_confirm_region not configured")
            
            if do_not_show_detected and confirm_tip_detected:
                detected_state = 'confirming_tips'
                logger.info(f"Game state detected: {detected_state} ('Do not show again' text AND 'confirm' text found)")
                if debug:
                    self._save_debug_screenshot(screenshot, f"detected_{detected_state}", annotations)
                return detected_state

            # Rule 16: Check for "Do not show again" or "confirm" text in tip_window_region (high priority)
            logger.info("Rule 16: Checking for do_not_show.png or confirm_button_2.png in tip_window_region (high priority)...")
            self._tip_window_do_not_show_pos = None
            self._tip_window_confirm_pos = None
            tip_window_bounds: Optional[Tuple[int, int, int, int]] = None

            tip_window_region = self._get_region('tip_window_region')
            if tip_window_region:
                x1, y1, x2, y2 = tip_window_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))

                if x2 > x1 and y2 > y1:
                    tip_window_bounds = (x1, y1, x2, y2)
                    region_image = screenshot[y1:y2, x1:x2]

                    do_not_show_result = self.template_matcher.find_template(
                        region_image,
                        "do_not_show.png",
                        multi_scale=True,
                        confidence=do_not_show_template_confidence
                    )
                    if do_not_show_result:
                        rel_x, rel_y, rel_conf = do_not_show_result
                        abs_pos = (x1 + int(rel_x), y1 + int(rel_y))
                        self._tip_window_do_not_show_pos = abs_pos
                        logger.info(
                            f"  - do_not_show.png detected in tip_window_region at ({abs_pos[0]}, {abs_pos[1]}) with confidence {rel_conf:.3f}"
                        )
                        if debug:
                            annotations['templates'].append({
                                'name': 'do_not_show.png',
                                'position': [abs_pos[0], abs_pos[1]],
                                'confidence': float(rel_conf),
                                'size': [0, 0]
                            })
                    else:
                        logger.info("  - do_not_show.png not detected in tip_window_region")

                    confirm_button_result = self.template_matcher.find_template(
                        region_image,
                        "confirm_button_2.png",
                        multi_scale=True,
                        confidence=confirm_button_template_confidence
                    )
                    if confirm_button_result:
                        rel_x, rel_y, rel_conf = confirm_button_result
                        abs_pos = (x1 + int(rel_x), y1 + int(rel_y))
                        self._tip_window_confirm_pos = abs_pos
                        logger.info(
                            f"  - confirm_button_2.png detected in tip_window_region at ({abs_pos[0]}, {abs_pos[1]}) with confidence {rel_conf:.3f}"
                        )
                        if debug:
                            annotations['templates'].append({
                                'name': 'confirm_button_2.png',
                                'position': [abs_pos[0], abs_pos[1]],
                                'confidence': float(rel_conf),
                                'size': [0, 0]
                            })
                    else:
                        logger.info("  - confirm_button_2.png not detected in tip_window_region")
                else:
                    logger.debug("Invalid tip_window_region coordinates")
            else:
                logger.debug("tip_window_region not configured")

            if self._tip_window_do_not_show_pos and self._tip_window_confirm_pos:
                detected_state = 'confirming_tips_window'
                logger.info("Game state detected: confirming_tips_window (Rule 16)")
                if debug and tip_window_bounds:
                    self._save_debug_screenshot(screenshot, "detected_confirming_tips_window", annotations)
                return detected_state

            if self._tip_window_do_not_show_pos or self._tip_window_confirm_pos:
                if not self._tip_window_do_not_show_pos and tip_window_bounds:
                    center_x = (tip_window_bounds[0] + tip_window_bounds[2]) // 2
                    center_y = (tip_window_bounds[1] + tip_window_bounds[3]) // 2
                    self._tip_window_do_not_show_pos = (center_x, center_y)
                if not self._tip_window_confirm_pos and tip_window_bounds:
                    center_x = (tip_window_bounds[0] + tip_window_bounds[2]) // 2
                    center_y = (tip_window_bounds[1] + tip_window_bounds[3]) // 2
                    self._tip_window_confirm_pos = (center_x, center_y)

                detected_state = 'confirming_tips_window_partial'
                logger.info("Game state detected: confirming_tips_window_partial (Rule 16 partial)")
                if debug and tip_window_bounds:
                    self._save_debug_screenshot(screenshot, "detected_confirming_tips_window_partial", annotations)
                return detected_state

            # Rule 14: Check for "Do not show again" text AND "confirm" text in tip_window_region
            logger.info("Rule 14: Checking for 'Do not show again' text AND 'confirm' text in tip_window_region...")

            # Reset stored positions before detection
            self._tip_window_do_not_show_pos = None
            self._tip_window_confirm_pos = None

            do_not_show_pos = None
            confirm_pos = None

            tip_window_region = self._get_region('tip_window_region')
            if tip_window_region:
                x1, y1, x2, y2 = tip_window_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))

                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]

                    # Detect "Do not show again" text position
                    do_not_show_pos = self._detect_do_not_show_again_text_position(region_image)
                    if do_not_show_pos:
                        # Convert relative position to absolute position
                        do_not_show_pos = (x1 + do_not_show_pos[0], y1 + do_not_show_pos[1])
                        logger.info(f"  - 'Do not show again' text detected in tip_window_region at ({do_not_show_pos[0]}, {do_not_show_pos[1]})")
                        if debug:
                            annotations['text'].append({
                                'text': 'Do not show again',
                                'position': [do_not_show_pos[0], do_not_show_pos[1]],
                                'confidence': 1.0
                            })
                    else:
                        logger.info("  - 'Do not show again' text not detected in tip_window_region")

                    # Detect "confirm" text position
                    confirm_pos = self._detect_confirm_text_position(region_image)
                    if confirm_pos:
                        # Convert relative position to absolute position
                        confirm_pos = (x1 + confirm_pos[0], y1 + confirm_pos[1])
                        logger.info(f"  - 'confirm' text detected in tip_window_region at ({confirm_pos[0]}, {confirm_pos[1]})")
                        if debug:
                            annotations['text'].append({
                                'text': 'confirm',
                                'position': [confirm_pos[0], confirm_pos[1]],
                                'confidence': 1.0
                            })
                    else:
                        logger.info("  - 'confirm' text not detected in tip_window_region")
                else:
                    logger.debug("Invalid tip_window_region coordinates")
            else:
                logger.debug("tip_window_region not configured")

            if do_not_show_pos and confirm_pos:
                detected_state = 'confirming_tips_window'
                logger.info(f"Game state detected: {detected_state} ('Do not show again' text AND 'confirm' text found in tip_window_region)")
                # Store positions for later use in automation
                self._tip_window_do_not_show_pos = do_not_show_pos
                self._tip_window_confirm_pos = confirm_pos
                if debug:
                    self._save_debug_screenshot(screenshot, f"detected_{detected_state}", annotations)
                return detected_state
            
            # Rule 15: Check for "Accept" text in source_accept_button_region
            logger.info("Rule 15: Checking for 'Accept' text in source_accept_button_region...")
            self._source_accept_button_pos = None

            source_accept_region = self._get_region('source_accept_button_region')
            if source_accept_region:
                x1, y1, x2, y2 = source_accept_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))

                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    accept_text_detected = self._detect_accept_text(region_image)

                    if accept_text_detected:
                        tap_x = (x1 + x2) // 2
                        tap_y = (y1 + y2) // 2
                        self._source_accept_button_pos = (tap_x, tap_y)
                        logger.info(f"  - 'Accept' text detected in source_accept_button_region at ({tap_x}, {tap_y})")
                        if debug:
                            annotations['text'].append({
                                'text': 'Accept',
                                'position': [tap_x, tap_y],
                                'confidence': 1.0
                            })

                        detected_state = 'accept_button'
                        logger.info("Game state detected: accept_button ('Accept' text found in source_accept_button_region)")
                        if debug:
                            self._save_debug_screenshot(screenshot, "detected_accept_button", annotations)
                        return detected_state
                    else:
                        logger.info("  - 'Accept' text not detected in source_accept_button_region")
                else:
                    logger.debug("Invalid source_accept_button_region coordinates")
            else:
                logger.debug("source_accept_button_region not configured")
            
            # Rule 17: Check for "Claim" text in source_claim_button_region
            logger.info("Rule 17: Checking for 'Claim' text in source_claim_button_region...")
            self._source_claim_button_pos = None

            source_claim_region = self._get_region('source_claim_button_region')
            if source_claim_region:
                x1, y1, x2, y2 = source_claim_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))

                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    claim_text_detected = self._detect_claim_text(region_image)

                    if claim_text_detected:
                        tap_x = (x1 + x2) // 2
                        tap_y = (y1 + y2) // 2
                        self._source_claim_button_pos = (tap_x, tap_y)
                        logger.info(f"  - 'Claim' text detected in source_claim_button_region at ({tap_x}, {tap_y})")
                        if debug:
                            annotations['text'].append({
                                'text': 'Claim',
                                'position': [tap_x, tap_y],
                                'confidence': 1.0
                            })

                        if self.adb and hasattr(self.adb, 'tap'):
                            if self.adb.tap(int(tap_x), int(tap_y)):
                                logger.info("Rule 17: Successfully tapped 'Claim' button in source_claim_button_region")
                            else:
                                logger.warning("Rule 17: Failed to tap 'Claim' button in source_claim_button_region")
                        else:
                            logger.warning("Rule 17: ADB tap not available - cannot tap 'Claim' button")

                        self._source_claim_button_pos = None

                        detected_state = 'claim_button'
                        logger.info("Game state detected: claim_button ('Claim' text found and tapped in source_claim_button_region)")
                        if debug:
                            self._save_debug_screenshot(screenshot, "detected_claim_button", annotations)
                        return detected_state
                    else:
                        logger.info("  - 'Claim' text not detected in source_claim_button_region")
                else:
                    logger.debug("Invalid source_claim_button_region coordinates")
            else:
                logger.debug("source_claim_button_region not configured")
            
            # Rule 10: Check for short_button.png AND "?" text in confirm_description_region
            logger.info("Rule 10: Checking for short_button.png AND '?' text in confirm_description_region...")
            
            # Check for short_button.png
            short_button_result = self.template_matcher.find_template(
                screenshot,
                "short_button.png",
                multi_scale=True,
                confidence=0.7
            )
            if short_button_result:
                teleport_short_button_detected = True
                x, y, confidence = short_button_result
                logger.info(f"  - short_button.png detected at ({x}, {y}) with confidence {confidence:.3f}")
                if debug:
                    annotations['templates'].append({
                        'name': 'short_button',
                        'position': [int(x), int(y)],
                        'confidence': confidence,
                        'size': [0, 0]
                    })
            else:
                logger.info(f"  - short_button.png not detected")
            
            # Check for "?" text in confirm_description_region
            confirm_desc_region = self._get_region('confirm_description_region')
            if confirm_desc_region:
                x1, y1, x2, y2 = confirm_desc_region
                screen_width = screenshot.shape[1]
                screen_height = screenshot.shape[0]
                x1 = max(0, min(x1, screen_width))
                y1 = max(0, min(y1, screen_height))
                x2 = max(0, min(x2, screen_width))
                y2 = max(0, min(y2, screen_height))
                
                if x2 > x1 and y2 > y1:
                    region_image = screenshot[y1:y2, x1:x2]
                    teleport_text_detected = self._detect_teleport_text(region_image)
                    if teleport_text_detected:
                        logger.info(f"  - '?' character detected in confirm_description_region")
                        if debug:
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            annotations['text'].append({
                                'text': '?',
                                'position': [center_x, center_y],
                                'confidence': 1.0
                            })
                    else:
                        logger.info(f"  - '?' character not detected in confirm_description_region")
                else:
                    logger.debug("Invalid confirm_description_region coordinates")
            else:
                logger.debug("confirm_description_region not configured")
            
            if teleport_short_button_detected and teleport_text_detected:
                # Teleporting detected - tap confirm button
                logger.info(f"Rule 10: Teleporting detected (short_button.png AND '?' character found) - tapping confirm button...")
                
                # Get teleport confirm button region from config
                teleport_confirm_region = self._get_region('teleport_confirm_button_region')
                
                tap_success = False
                if teleport_confirm_region and self.adb and hasattr(self.adb, 'tap'):
                    x1, y1, x2, y2 = teleport_confirm_region
                    tap_x = (x1 + x2) // 2
                    tap_y = (y1 + y2) // 2
                    
                    screen_width = screenshot.shape[1]
                    screen_height = screenshot.shape[0]
                    if 0 <= tap_x < screen_width and 0 <= tap_y < screen_height:
                        logger.info(f"[RULE 10] Tapping teleport confirm button at ({tap_x}, {tap_y})")
                        tap_success = self.adb.tap(int(tap_x), int(tap_y))
                        if tap_success:
                            logger.info(f"[RULE 10] Successfully tapped teleport confirm button")
                        else:
                            logger.warning(f"[RULE 10] Failed to tap teleport confirm button")
                    else:
                        logger.warning(f"[RULE 10] Invalid teleport_confirm_button_region coordinates: ({tap_x}, {tap_y})")
                else:
                    if not teleport_confirm_region:
                        logger.warning("[RULE 10] teleport_confirm_button_region not configured")
                    if not (self.adb and hasattr(self.adb, 'tap')):
                        logger.warning("[RULE 10] ADB manager not available or tap method not found")
                
                # Set detailed_game_state to 'auto_questing' after successfully tapping confirm button
                if tap_success:
                    self._detected_detailed_state = 'auto_questing'
                    logger.info(f"Rule 10: Successfully tapped confirm button - setting detailed_game_state to 'auto_questing'")
                else:
                    # If tap failed, set to 'teleporting' to indicate teleporting was detected but action not completed
                    self._detected_detailed_state = 'teleporting'
                    logger.info(f"Rule 10: Failed to tap confirm button - setting detailed_game_state to 'teleporting'")
                if debug:
                    self._save_debug_screenshot(screenshot, f"detected_teleporting", annotations)
                    self._save_template_images(['short_button.png'])
                # Don't return here - continue with normal detection to return 'playing' state
            
            # Rule 11: Check for "Resurrect" text in resurrect_button_region (only if in playing state)
            if self._has_entered_playing:
                logger.info("Rule 11: Checking for 'Resurrect' text in resurrect_button_region...")
                
                resurrect_button_region = self._get_region('resurrect_button_region')
                if resurrect_button_region:
                    x1, y1, x2, y2 = resurrect_button_region
                    screen_width = screenshot.shape[1]
                    screen_height = screenshot.shape[0]
                    x1 = max(0, min(x1, screen_width))
                    y1 = max(0, min(y1, screen_height))
                    x2 = max(0, min(x2, screen_width))
                    y2 = max(0, min(y2, screen_height))
                    
                    if x2 > x1 and y2 > y1:
                        region_image = screenshot[y1:y2, x1:x2]
                        resurrect_text_detected = self._detect_resurrect_text(region_image)
                        if resurrect_text_detected:
                            logger.info(f"  - 'Resurrect' text detected in resurrect_button_region")
                            if debug:
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                annotations['text'].append({
                                    'text': 'Resurrect',
                                    'position': [center_x, center_y],
                                    'confidence': 1.0
                                })
                        else:
                            logger.info(f"  - 'Resurrect' text not detected in resurrect_button_region")
                    else:
                        logger.debug("Invalid resurrect_button_region coordinates")
                else:
                    logger.debug("resurrect_button_region not configured")
                
                if resurrect_text_detected:
                    # Resurrect text detected - tap resurrect button and set detailed_game_state accordingly
                    logger.info(f"Rule 11: Resurrect text detected - tapping resurrect button...")
                    
                    tap_success = False
                    # Tap resurrect_button_region
                    if resurrect_button_region and self.adb and hasattr(self.adb, 'tap'):
                        x1, y1, x2, y2 = resurrect_button_region
                        tap_x = (x1 + x2) // 2
                        tap_y = (y1 + y2) // 2
                        
                        screen_width = screenshot.shape[1]
                        screen_height = screenshot.shape[0]
                        if 0 <= tap_x < screen_width and 0 <= tap_y < screen_height:
                            logger.info(f"[RULE 11] Tapping resurrect button at ({tap_x}, {tap_y})")
                            tap_success = self.adb.tap(int(tap_x), int(tap_y))
                            if tap_success:
                                logger.info(f"[RULE 11] Successfully tapped resurrect button")
                            else:
                                logger.warning(f"[RULE 11] Failed to tap resurrect button")
                        else:
                            logger.warning(f"[RULE 11] Invalid resurrect_button_region coordinates: ({tap_x}, {tap_y})")
                    else:
                        if not resurrect_button_region:
                            logger.warning("[RULE 11] resurrect_button_region not configured")
                        if not (self.adb and hasattr(self.adb, 'tap')):
                            logger.warning("[RULE 11] ADB manager not available or tap method not found")
                    
                    # Set detailed_game_state based on tap success
                    if tap_success:
                        # After successfully tapping resurrect button, set to 'purchasing'
                        self._detected_detailed_state = 'purchasing'
                        logger.info(f"Rule 11: Successfully tapped resurrect button - setting detailed_game_state to 'purchasing'")
                        if debug:
                            self._save_debug_screenshot(screenshot, f"detected_purchasing", annotations)
                    else:
                        # If tap failed, set to 'dead' to indicate dead state detected but action not completed
                        self._detected_detailed_state = 'dead'
                        logger.info(f"Rule 11: Failed to tap resurrect button - setting detailed_game_state to 'dead'")
                        if debug:
                            self._save_debug_screenshot(screenshot, f"detected_dead", annotations)
            
            if self._has_entered_playing:
                logger.info("Playing state already confirmed; skipping further detection checks.")
                return 'playing'

            if not self._has_entered_playing:
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
            else:
                logger.info("Skipping select_server/select_character checks (already entered playing state).")
            
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
                if not self._has_entered_playing:
                    logger.debug("Marking playing state as reached; future detections will skip pre-play checks.")
                self._has_entered_playing = True
                
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
            logger.info(f"    - unlock_screen_region: {unlock_screen_detected}")
            logger.info(f"    - claim_reward_button: {claim_reward_detected}")
            logger.info(f"    - accepting_reward_button: {accepting_reward_detected}")
            logger.info(f"    - close_cross: {close_cross_detected}")
            logger.info(f"    - do_not_show_again: {do_not_show_detected}")
            logger.info(f"    - confirm_tip: {confirm_tip_detected}")
            logger.info(f"    - teleport_short_button: {teleport_short_button_detected}")
            logger.info(f"    - teleport_text: {teleport_text_detected}")
            logger.info(f"    - resurrect_text: {resurrect_text_detected}")
            logger.info(f"    - short_button: {fight_button_detected}")
            logger.info(f"    - skip_button: {skip_button_detected}")
            logger.info(f"  Result: {detected_state}")

            if self._has_entered_playing:
                logger.info("Previously detected playing state; forcing detected state to 'playing'.")
                return 'playing'
            
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
    
    def _save_image_rgb(self, image: np.ndarray, filepath: str) -> bool:
        """
        Save image with correct RGB color format (converts BGR to RGB if needed)
        
        Args:
            image: OpenCV image array (BGR format)
            filepath: Path to save the image
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert BGR to RGB before saving (OpenCV uses BGR, but image viewers expect RGB)
            rgb_image = image.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2RGB'):
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Manual BGR to RGB conversion
                    rgb_image = image[:, :, ::-1]
            
            # Save using PIL (which expects RGB format)
            from PIL import Image  # type: ignore
            if len(rgb_image.shape) == 3:
                pil_image = Image.fromarray(rgb_image)
            else:
                pil_image = Image.fromarray(rgb_image)
            
            pil_image.save(str(filepath))
            return True
        except Exception as e:
            logger.warning(f"Failed to save image with RGB conversion: {e}, trying cv2.imwrite fallback")
            # Fallback: try cv2.imwrite (but colors will be reversed in image viewers)
            try:
                if hasattr(cv2, 'imwrite'):
                    cv2.imwrite(str(filepath), image)
                    return True
            except Exception as e2:
                logger.error(f"Failed to save image with both PIL and cv2: {e2}")
                return False
    
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
            
            # Convert BGR to RGB before saving (OpenCV uses BGR, but image viewers expect RGB)
            # This ensures saved images display with correct colors
            rgb_screenshot = annotated_screenshot.copy()
            if len(annotated_screenshot.shape) == 3 and annotated_screenshot.shape[2] == 3:
                # Convert BGR to RGB
                if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2RGB'):
                    rgb_screenshot = cv2.cvtColor(annotated_screenshot, cv2.COLOR_BGR2RGB)
                else:
                    # Manual BGR to RGB conversion
                    rgb_screenshot = annotated_screenshot[:, :, ::-1]
            
            # Save using PIL (which expects RGB format)
            try:
                from PIL import Image  # type: ignore
                if len(rgb_screenshot.shape) == 3:
                    pil_image = Image.fromarray(rgb_screenshot)
                else:
                    pil_image = Image.fromarray(rgb_screenshot)
                
                pil_image.save(str(filename))
                logger.info(f"Debug screenshot saved: {filename}")
            except Exception as e:
                logger.warning(f"Failed to save debug screenshot with PIL: {e}")
                # Fallback: try cv2.imwrite (but colors will be reversed in image viewers)
                try:
                    if hasattr(cv2, 'imwrite'):
                        cv2.imwrite(str(filename), annotated_screenshot)
                        logger.warning(f"Saved with cv2.imwrite (colors may appear reversed in image viewers): {filename}")
                except Exception as e2:
                    logger.error(f"Failed to save debug screenshot with both PIL and cv2: {e2}")
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
            from PIL import ImageDraw, ImageFont  # type: ignore
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
            pass
    
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
        """Load region configurations from JSON file (shared across all devices)"""
        try:
            if device_id:
                self.device_id = device_id
            
            # Always use shared regions.json file
            region_file = Path('config') / 'regions.json'
            logger.debug(f"Attempting to load regions from: {region_file}")
            
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
    
    def has_entered_playing_state(self) -> bool:
        """Return True if we've detected the playing state at least once this session."""
        return self._has_entered_playing
    
    def get_detected_detailed_state(self) -> Optional[str]:
        """
        Get and clear the detected detailed game state.
        This is used for states that should set detailed_game_state but not change actual_game_state.
        
        Returns:
            The detected detailed state (e.g., 'teleporting') or None
        """
        state = self._detected_detailed_state
        self._detected_detailed_state = None  # Clear after reading
        return state
    
    def detect_quests_in_region(self, screenshot: np.ndarray) -> List[Tuple[int, int, float, Tuple[int, int]]]:
        """Find quest buttons in the configured quests region and return matches sorted top-down."""
        matches: List[Tuple[int, int, float, Tuple[int, int]]] = []
        quests_region = self._get_region('quests')
        if not quests_region:
            logger.debug("Quests region not configured; skipping quest detection.")
            return matches

        x1, y1, x2, y2 = quests_region
        screen_width = screenshot.shape[1]
        screen_height = screenshot.shape[0]
        x1 = max(0, min(x1, screen_width))
        y1 = max(0, min(y1, screen_height))
        x2 = max(0, min(x2, screen_width))
        y2 = max(0, min(y2, screen_height))
        if x2 <= x1 or y2 <= y1:
            logger.debug("Quests region coordinates invalid; skipping quest detection.")
            return matches

        region = screenshot[y1:y2, x1:x2]
        template_names = [
            "quest_button.png",
            "quest_button_highlight.png",
            "quest_button_alt.png"
        ]

        for template_name in template_names:
            template = self.template_matcher.load_template(template_name)
            if template is None:
                logger.debug(f"Quest template {template_name} missing; skipping.")
                continue

            detections = self.template_matcher.find_all_templates(
                region,
                template_name,
                confidence=0.7,
            )

            for detection in detections or []:
                det_x, det_y, det_confidence = detection
                matches.append((
                    int(x1 + det_x),
                    int(y1 + det_y),
                    float(det_confidence),
                    (int(template.shape[1]), int(template.shape[0]))
                ))

        matches.sort(key=lambda item: item[1])
        return matches

    def tap_agent_quest_region_center(self, screenshot: Optional[np.ndarray] = None) -> bool:
        """Tap the center of the configured agent quest region."""
        agent_region = self._get_region('agent_quest_region')
        if not agent_region:
            logger.debug("agent_quest_region not configured; cannot tap center.")
            return False

        x1, y1, x2, y2 = agent_region
        if screenshot is not None:
            screen_width = screenshot.shape[1]
            screen_height = screenshot.shape[0]
            x1 = max(0, min(x1, screen_width))
            y1 = max(0, min(y1, screen_height))
            x2 = max(0, min(x2, screen_width))
            y2 = max(0, min(y2, screen_height))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid agent_quest_region coordinates for center tap: ({x1}, {y1}, {x2}, {y2})")
            return False

        tap_x = (x1 + x2) // 2
        tap_y = (y1 + y2) // 2
        logger.info(f"Tapping agent quest region center at ({tap_x}, {tap_y})")

        if self.adb and hasattr(self.adb, 'tap'):
            success = self.adb.tap(int(tap_x), int(tap_y))
            if success:
                logger.info("Successfully tapped agent quest region center")
                return True
            logger.warning("Failed to tap agent quest region center")
            return False

        logger.warning("ADB manager not available or tap method not found for agent quest region center")
        return False

    # Rule 15: Check for accept_button.png
    pass

    def _detect_skip_text(self, region_image: np.ndarray) -> bool:
        """
        Detect "Skip" text in a region using Tesseract OCR.

        Args:
            region_image: OpenCV image array (cropped region)

        Returns:
            True if "Skip" text is detected, False otherwise
        """
        try:
            from ..utils.tesseract_ocr import get_tesseract_reader
            ocr_reader = get_tesseract_reader(lang='eng')
            if ocr_reader is None:
                return False

            results = None
            try:
                results = ocr_reader.readtext(region_image, detail=1, paragraph=False)

                skip_keywords = ['skip']
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    for keyword in skip_keywords:
                        if keyword in text_lower:
                            logger.info(f"Detected 'Skip' text: '{text}' (confidence: {confidence:.3f})")
                            return True
                return False
            finally:
                if results is not None:
                    del results
        except Exception as e:
            logger.debug(f"Error detecting Skip text: {e}")
            return False