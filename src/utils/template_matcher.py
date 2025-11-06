"""
Template Matcher - Finds game UI elements (buttons, icons) in screenshots using template matching
Supports multi-scale matching and OCR text extraction
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import os

from ..utils.logger import get_logger
from ..utils.config import config_manager

def _bgr_to_gray_cv2(image: np.ndarray) -> np.ndarray:
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

def _resize_cv2(image: np.ndarray, size: Tuple[int, int], interpolation: int = None) -> np.ndarray:
    """Resize image - fallback if cv2.resize doesn't exist"""
    if hasattr(cv2, 'resize'):
        # Map interpolation constants if needed
        if interpolation is None:
            interpolation = cv2.INTER_CUBIC if hasattr(cv2, 'INTER_CUBIC') else 3
        elif not isinstance(interpolation, int):
            # If it's an attribute, try to get its value
            if hasattr(cv2, str(interpolation).split('.')[-1]):
                interp_name = str(interpolation).split('.')[-1]
                interpolation = getattr(cv2, interp_name, 3)
            else:
                interpolation = 3  # Default to cubic
        return cv2.resize(image, size, interpolation=interpolation)
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
        
        # Resize using PIL (LANCZOS is similar to INTER_CUBIC)
        resized_pil = pil_image.resize(size, Image.LANCZOS)
        
        # Convert back to numpy array
        resized = np.array(resized_pil)
        
        # Convert RGB back to BGR if needed
        if len(resized.shape) == 3:
            resized = resized[:, :, ::-1]
        
        return resized

# Try to import Tesseract OCR (optional)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = get_logger(__name__)


class TemplateMatcher:
    """
    Template matching utility for finding game UI elements in screenshots
    """
    # Class-level flag to track if OpenCV warning has been logged (only log once globally)
    _opencv_warning_logged_class = False
    
    def __init__(self, templates_dir: str = "assets/templates"):
        """
        Initialize template matcher
        
        Args:
            templates_dir: Directory containing template images
        """
        self.templates_dir = Path(templates_dir)
        self.config = config_manager.get_config()
        self.confidence_threshold = self.config.image_recognition.confidence_threshold
        
        # Parse template matching method from config
        # OpenCV template matching constants (as integers if not available as attributes)
        TM_CONSTANTS = {
            'TM_SQDIFF': 0,
            'TM_SQDIFF_NORMED': 1,
            'TM_CCORR': 2,
            'TM_CCORR_NORMED': 3,
            'TM_CCOEFF': 4,
            'TM_CCOEFF_NORMED': 5
        }
        
        method_str = self.config.image_recognition.template_matching_method
        method_name = method_str.split('.')[-1]
        
        # Try to get from cv2 attributes first, fallback to integer constants
        if hasattr(cv2, method_name):
            self.matching_method = getattr(cv2, method_name)
        elif method_name in TM_CONSTANTS:
            self.matching_method = TM_CONSTANTS[method_name]
        else:
            # Default to TM_CCOEFF_NORMED (value 5)
            self.matching_method = TM_CONSTANTS.get('TM_CCOEFF_NORMED', 5)
            logger.warning(f"Unknown template matching method: {method_name}, using TM_CCOEFF_NORMED")
        
        # Cache for loaded templates
        self.template_cache: Dict[str, np.ndarray] = {}
        
        # Multi-scale matching parameters
        self.scales = np.linspace(0.5, 2.0, 20)  # Default scales to try
        self.pad = 12  # Padding around detected region for OCR
        
        # OCR reader (lazy initialization - using Tesseract instead of EasyOCR)
        self._ocr_reader = None
        
        # Track if we've already warned about missing OpenCV functions (to avoid spam)
        self._opencv_warning_logged = False
        
        # Check if OpenCV has required functions
        self._opencv_available = (
            hasattr(cv2, 'matchTemplate') and 
            hasattr(cv2, 'minMaxLoc') and 
            hasattr(cv2, 'resize') and
            hasattr(cv2, 'imread')
        )
        
        # Only log warning once globally (class-level check)
        if not self._opencv_available and not TemplateMatcher._opencv_warning_logged_class:
            logger.error(
                "WARNING: OpenCV installation appears corrupted - template matching will not work!\n"
                "Required functions missing: matchTemplate, minMaxLoc, resize, or imread\n"
                "Please reinstall OpenCV with:\n"
                "  pip uninstall opencv-python opencv-python-headless -y\n"
                "  pip install --force-reinstall --no-cache-dir opencv-python"
            )
            TemplateMatcher._opencv_warning_logged_class = True
        
        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Template matcher initialized with templates directory: {self.templates_dir}")
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract not available. OCR features will be disabled. Install with: pip install pytesseract")
    
    def load_template(self, template_name: str) -> Optional[np.ndarray]:
        """
        Load a template image from disk
        
        Args:
            template_name: Name of the template file (e.g., 'attack_button.png')
            
        Returns:
            Template image as numpy array, or None if not found
        """
        # Check cache first
        if template_name in self.template_cache:
            return self.template_cache[template_name]
        
        # Try to load from templates directory
        template_path = self.templates_dir / template_name
        
        if not template_path.exists():
            logger.warning(f"Template not found: {template_path}")
            return None
        
        try:
            # Load template - try OpenCV first, fallback to PIL
            if hasattr(cv2, 'imread'):
                template = cv2.imread(str(template_path), cv2.IMREAD_COLOR if hasattr(cv2, 'IMREAD_COLOR') else None)
            else:
                # Fallback: Use PIL to load image and convert to OpenCV format
                from PIL import Image
                pil_image = Image.open(template_path)
                template = np.array(pil_image)
                if len(template.shape) == 3:
                    # Convert RGB to BGR for OpenCV compatibility
                    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_RGB2BGR'):
                        template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
                    else:
                        # Manual RGB to BGR conversion using numpy
                        template = template[:, :, ::-1]  # Reverse RGB channels to BGR
            if template is None:
                logger.error(f"Failed to load template: {template_path}")
                return None
            
            # Cache the template
            self.template_cache[template_name] = template
            logger.debug(f"Loaded template: {template_name}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            return None
    
    def find_template(self, 
                     screenshot: np.ndarray, 
                     template_name: str,
                     confidence: float = None,
                     region: Tuple[int, int, int, int] = None,
                     multi_scale: bool = True) -> Optional[Tuple[int, int, float]]:
        """
        Find a template in a screenshot using single-scale or multi-scale matching
        
        Args:
            screenshot: Screenshot image to search in
            template_name: Name of the template to find
            confidence: Minimum confidence threshold (uses config default if None)
            region: Optional region to search in (x, y, width, height)
            multi_scale: Whether to use multi-scale matching (default: True)
            
        Returns:
            Tuple of (center_x, center_y, confidence) if found, None otherwise
        """
        # Load template
        template = self.load_template(template_name)
        if template is None:
            return None
        
        # Use provided confidence or default
        threshold = confidence if confidence is not None else self.confidence_threshold
        
        # Extract region if specified
        if region:
            x, y, w, h = region
            search_area = screenshot[y:y+h, x:x+w]
            if search_area.size == 0:
                logger.warning(f"Invalid search region: {region}")
                return None
        else:
            search_area = screenshot
        
        try:
            if multi_scale:
                # Multi-scale template matching
                return self._find_template_multiscale(search_area, template, threshold, region)
            else:
                # Single-scale template matching
                return self._find_template_singlescale(search_area, template, threshold, region, template_name)
            
        except Exception as e:
            logger.error(f"Error matching template {template_name}: {e}")
            return None
    
    def find_template_best_match(self,
                                 screenshot: np.ndarray,
                                 template_name: str,
                                 region: Tuple[int, int, int, int] = None,
                                 multi_scale: bool = True) -> Optional[Tuple[int, int, float]]:
        """
        Find the best template match regardless of threshold (for debugging)
        
        Args:
            screenshot: Screenshot image to search in
            template_name: Name of the template to find
            region: Optional region to search in (x, y, width, height)
            multi_scale: Whether to use multi-scale matching
            
        Returns:
            Tuple of (center_x, center_y, confidence) of best match found, or None
        """
        # Load template
        template = self.load_template(template_name)
        if template is None:
            return None
        
        # Extract region if specified
        if region:
            x, y, w, h = region
            search_area = screenshot[y:y+h, x:x+w]
            if search_area.size == 0:
                logger.warning(f"Invalid search region: {region}")
                return None
        else:
            search_area = screenshot
        
        try:
            if multi_scale:
                # Use multi-scale matching but return best match regardless of threshold
                return self._find_template_multiscale_best(search_area, template, region)
            else:
                # Use single-scale matching but return best match regardless of threshold
                return self._find_template_singlescale_best(search_area, template, region, template_name)
        except Exception as e:
            logger.error(f"Error finding best match for template {template_name}: {e}")
            return None
    
    def _find_template_singlescale_best(self, search_area: np.ndarray, template: np.ndarray,
                                       region: Tuple[int, int, int, int], template_name: str) -> Optional[Tuple[int, int, float]]:
        """Single-scale template matching - returns best match regardless of threshold"""
        # Convert to grayscale for matching
        if len(search_area.shape) == 3:
            search_gray = _bgr_to_gray_cv2(search_area)
        else:
            search_gray = search_area
        
        if len(template.shape) == 3:
            template_gray = _bgr_to_gray_cv2(template)
        else:
            template_gray = template
        
        # Perform template matching
        if not self._opencv_available or not hasattr(cv2, 'matchTemplate'):
            return None
        
        try:
            result = cv2.matchTemplate(search_gray, template_gray, self.matching_method)
        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return None
        
        # Get the best match
        try:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        except Exception as e:
            logger.error(f"Error in minMaxLoc: {e}")
            return None
        
        # For TM_CCOEFF_NORMED, higher is better
        if self.matching_method in [0, 1]:  # TM_SQDIFF, TM_SQDIFF_NORMED
            confidence_score = 1 - min_val
            match_loc = min_loc
        else:
            confidence_score = max_val
            match_loc = max_loc
        
        # Calculate center position (always return best match regardless of threshold)
        template_h, template_w = template.shape[:2]
        
        if region:
            center_x = region[0] + match_loc[0] + template_w // 2
            center_y = region[1] + match_loc[1] + template_h // 2
        else:
            center_x = match_loc[0] + template_w // 2
            center_y = match_loc[1] + template_h // 2
        
        logger.info(f"Best match for {template_name}: location=({match_loc[0]}, {match_loc[1]}), center=({center_x}, {center_y}), confidence={confidence_score:.3f}")
        return (center_x, center_y, confidence_score)
    
    def _find_template_multiscale_best(self, search_area: np.ndarray, template: np.ndarray,
                                      region: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, float]]:
        """Multi-scale template matching - returns best match regardless of threshold"""
        # Convert to grayscale
        if len(search_area.shape) == 3:
            search_gray = _bgr_to_gray_cv2(search_area)
        else:
            search_gray = search_area
            
        if len(template.shape) == 3:
            template_gray = _bgr_to_gray_cv2(template)
        else:
            template_gray = template
        
        best_val = -1
        best_loc = None
        best_size = None
        best_scale = None
        
        # Check if OpenCV is available
        if not self._opencv_available or not hasattr(cv2, 'matchTemplate') or not hasattr(cv2, 'minMaxLoc'):
            return None
        
        logger.debug(f"Multi-scale matching (best match): trying {len(self.scales)} scales")
        
        for scale in self.scales:
            # Resize template
            new_size = (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale))
            interpolation = cv2.INTER_CUBIC if hasattr(cv2, 'INTER_CUBIC') else 3
            resized = _resize_cv2(template_gray, new_size, interpolation=interpolation)
            
            # Skip if resized template is larger than search area
            if resized.shape[0] > search_gray.shape[0] or resized.shape[1] > search_gray.shape[1]:
                continue
            
            # Match template
            try:
                res = cv2.matchTemplate(search_gray, resized, self.matching_method)
            except Exception as e:
                logger.error(f"Error in template matching at scale {scale}: {e}")
                continue
            
            try:
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
            except Exception as e:
                logger.error(f"Error in minMaxLoc at scale {scale}: {e}")
                continue
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_size = resized.shape[::-1]  # (width, height)
                best_scale = scale
        
        # Return best match regardless of threshold
        if best_loc is None:
            logger.warning("No match found at any scale")
            return None
        
        # Calculate center position
        w, h = best_size
        
        if region:
            center_x = region[0] + best_loc[0] + w // 2
            center_y = region[1] + best_loc[1] + h // 2
        else:
            center_x = best_loc[0] + w // 2
            center_y = best_loc[1] + h // 2
        
        logger.info(f"Best multi-scale match: location=({best_loc[0]}, {best_loc[1]}), center=({center_x}, {center_y}), confidence={best_val:.3f}, scale={best_scale:.2f}")
        return (center_x, center_y, best_val)
    
    def _find_template_singlescale(self, search_area: np.ndarray, template: np.ndarray,
                                   threshold: float, region: Tuple[int, int, int, int],
                                   template_name: str) -> Optional[Tuple[int, int, float]]:
        """Single-scale template matching"""
        # Convert to grayscale for matching
        if len(search_area.shape) == 3:
            search_gray = _bgr_to_gray_cv2(search_area)
        else:
            search_gray = search_area
        
        if len(template.shape) == 3:
            template_gray = _bgr_to_gray_cv2(template)
        else:
            template_gray = template
        
        # Perform template matching
        if not self._opencv_available or not hasattr(cv2, 'matchTemplate'):
            if not self._opencv_warning_logged:
                logger.debug("Skipping template matching - OpenCV functions not available (already warned at initialization)")
                self._opencv_warning_logged = True
            return None
        
        try:
            result = cv2.matchTemplate(search_gray, template_gray, self.matching_method)
        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return None
        
        # Get the best match
        try:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        except Exception as e:
            logger.error(f"Error in minMaxLoc: {e}")
            return None
        
        # For TM_CCOEFF_NORMED, higher is better
        # Use integer values for comparison (TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCOEFF_NORMED=5)
        if self.matching_method in [0, 1]:  # TM_SQDIFF, TM_SQDIFF_NORMED
            confidence_score = 1 - min_val
            match_loc = min_loc
        else:
            confidence_score = max_val
            match_loc = max_loc
        
        # Check if confidence meets threshold
        if confidence_score < threshold:
            logger.warning(f"Template {template_name} found but confidence {confidence_score:.3f} < threshold {threshold} (match location: {match_loc})")
            logger.info(f"Best match for {template_name}: location={match_loc}, confidence={confidence_score:.3f}, threshold={threshold}")
            return None
        
        # Calculate center position
        template_h, template_w = template.shape[:2]
        
        if region:
            # Adjust coordinates for region
            center_x = region[0] + match_loc[0] + template_w // 2
            center_y = region[1] + match_loc[1] + template_h // 2
        else:
            center_x = match_loc[0] + template_w // 2
            center_y = match_loc[1] + template_h // 2
        
        logger.debug(f"Found template {template_name} at ({center_x}, {center_y}) with confidence {confidence_score:.2f}")
        
        return (center_x, center_y, confidence_score)
    
    def _find_template_multiscale(self, search_area: np.ndarray, template: np.ndarray,
                                 threshold: float, region: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, float]]:
        """Multi-scale template matching (like the sample code)"""
        # Convert to grayscale
        if len(search_area.shape) == 3:
            search_gray = _bgr_to_gray_cv2(search_area)
        else:
            search_gray = search_area
            
        if len(template.shape) == 3:
            template_gray = _bgr_to_gray_cv2(template)
        else:
            template_gray = template
        
        best_val = -1
        best_loc = None
        best_size = None
        
        # Check if OpenCV is available before starting multi-scale matching
        if not self._opencv_available or not hasattr(cv2, 'matchTemplate') or not hasattr(cv2, 'minMaxLoc'):
            if not self._opencv_warning_logged:
                logger.debug("Skipping multi-scale template matching - OpenCV functions not available (already warned at initialization)")
                self._opencv_warning_logged = True
            return None
        
        logger.debug(f"Multi-scale matching: trying {len(self.scales)} scales")
        
        for scale in self.scales:
            # Resize template
            new_size = (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale))
            interpolation = cv2.INTER_CUBIC if hasattr(cv2, 'INTER_CUBIC') else 3
            resized = _resize_cv2(
                template_gray,
                new_size,
                interpolation=interpolation,
            )
            
            # Skip if resized template is larger than search area
            if resized.shape[0] > search_gray.shape[0] or resized.shape[1] > search_gray.shape[1]:
                continue
            
            # Match template
            try:
                res = cv2.matchTemplate(search_gray, resized, self.matching_method)
            except Exception as e:
                logger.error(f"Error in template matching at scale {scale}: {e}")
                continue
            
            try:
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
            except Exception as e:
                logger.error(f"Error in minMaxLoc at scale {scale}: {e}")
                continue
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_size = resized.shape[::-1]  # (width, height)
        
        # Check if confidence meets threshold
        if best_val < threshold or best_loc is None:
            logger.warning(f"Template match found but confidence {best_val:.3f} < threshold {threshold} (best match was at {best_loc})")
            # Log even if below threshold to help debugging
            if best_loc is not None:
                logger.info(f"Best match found at location {best_loc} with confidence {best_val:.3f} (threshold: {threshold})")
            return None
        
        # Calculate center position
        w, h = best_size
        
        if region:
            # Adjust coordinates for region
            center_x = region[0] + best_loc[0] + w // 2
            center_y = region[1] + best_loc[1] + h // 2
        else:
            center_x = best_loc[0] + w // 2
            center_y = best_loc[1] + h // 2
        
        logger.debug(f"Multi-scale match found at ({center_x}, {center_y}) with confidence {best_val:.3f} at scale {best_size}")
        
        return (center_x, center_y, best_val)
    
    def find_all_templates(self,
                          screenshot: np.ndarray,
                          template_name: str,
                          confidence: float = None,
                          max_results: int = 10) -> List[Tuple[int, int, float]]:
        """
        Find all occurrences of a template in a screenshot
        
        Args:
            screenshot: Screenshot image to search in
            template_name: Name of the template to find
            confidence: Minimum confidence threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples (center_x, center_y, confidence) for each match
        """
        # Load template
        template = self.load_template(template_name)
        if template is None:
            return []
        
        threshold = confidence if confidence is not None else self.confidence_threshold
        
        try:
            # Perform template matching
            if not hasattr(cv2, 'matchTemplate'):
                logger.error("cv2.matchTemplate not available - OpenCV installation appears corrupted")
                return []
            try:
                result = cv2.matchTemplate(screenshot, template, self.matching_method)
            except Exception as e:
                logger.error(f"Error in template matching: {e}")
                return []
            
            # Find all matches above threshold
            # Use integer values for comparison (TM_SQDIFF=0, TM_SQDIFF_NORMED=1)
            if self.matching_method in [0, 1]:  # TM_SQDIFF, TM_SQDIFF_NORMED
                locations = np.where(result <= (1 - threshold))
                values = 1 - result[locations]
            else:
                locations = np.where(result >= threshold)
                values = result[locations]
            
            # Convert to list of (x, y, confidence)
            matches = []
            template_h, template_w = template.shape[:2]
            
            for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                x, y = pt
                confidence_score = values[y, x]
                
                # Calculate center
                center_x = x + template_w // 2
                center_y = y + template_h // 2
                
                matches.append((center_x, center_y, confidence_score))
            
            # Sort by confidence (highest first) and limit results
            matches.sort(key=lambda m: m[2], reverse=True)
            
            # Remove overlapping matches (keep highest confidence)
            filtered_matches = []
            for match in matches[:max_results]:
                x, y, conf = match
                # Check if this match overlaps with any existing match
                is_duplicate = False
                for existing in filtered_matches:
                    ex, ey, _ = existing
                    distance = np.sqrt((x - ex)**2 + (y - ey)**2)
                    # If within template size, consider it duplicate
                    if distance < max(template_w, template_h):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_matches.append(match)
            
            logger.debug(f"Found {len(filtered_matches)} matches for template {template_name}")
            return filtered_matches
            
        except Exception as e:
            logger.error(f"Error finding all templates {template_name}: {e}")
            return []
    
    def find_any_template(self,
                         screenshot: np.ndarray,
                         template_names: List[str],
                         confidence: float = None) -> Optional[Tuple[str, int, int, float]]:
        """
        Find any of multiple templates in a screenshot
        
        Args:
            screenshot: Screenshot image to search in
            template_names: List of template names to try
            confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (template_name, center_x, center_y, confidence) if found, None otherwise
        """
        best_match = None
        best_confidence = 0
        
        for template_name in template_names:
            result = self.find_template(screenshot, template_name, confidence)
            if result:
                x, y, conf = result
                if conf > best_confidence:
                    best_confidence = conf
                    best_match = (template_name, x, y, conf)
        
        return best_match
    
    def clear_cache(self):
        """Clear the template cache"""
        self.template_cache.clear()
        logger.debug("Template cache cleared")
    
    def list_templates(self) -> List[str]:
        """
        List all available template files
        
        Returns:
            List of template file names
        """
        templates = []
        if self.templates_dir.exists():
            for file in self.templates_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    templates.append(file.name)
        
        return sorted(templates)
    
    def _get_ocr_reader(self):
        """Get shared Tesseract OCR reader (to avoid multiple instances)"""
        if not TESSERACT_AVAILABLE:
            return None
        
        # Use shared Tesseract OCR reader singleton
        from ..utils.tesseract_ocr import get_tesseract_reader
        return get_tesseract_reader(lang='eng')
    
    def find_template_and_extract_text(self,
                                      screenshot: np.ndarray,
                                      template_name: str,
                                      confidence: float = None,
                                      region: Tuple[int, int, int, int] = None,
                                      multi_scale: bool = True) -> Optional[Dict[str, Any]]:
        """
        Find a template and extract text from the detected region (like the sample code)
        
        Args:
            screenshot: Screenshot image to search in
            template_name: Name of the template to find
            confidence: Minimum confidence threshold
            region: Optional region to search in
            multi_scale: Whether to use multi-scale matching
            
        Returns:
            Dictionary with:
                - position: (center_x, center_y)
                - confidence: match confidence
                - text: extracted text (if OCR available)
                - text_confidence: average OCR confidence
                - bbox: bounding box (x1, y1, x2, y2) of detected region
        """
        # Find template first
        result = self.find_template(screenshot, template_name, confidence, region, multi_scale)
        
        if result is None:
            return None
        
        center_x, center_y, match_confidence = result
        
        # Load template to get its size
        template = self.load_template(template_name)
        if template is None:
            return None
        
        # Get detected region coordinates (with padding)
        template_h, template_w = template.shape[:2]
        
        # If multi-scale was used, we need to get the actual detected size
        # For now, use template size
        if multi_scale:
            # We need to recalculate the actual size from the match
            # This is a simplified version - in practice you'd store the best_size
            detected_w, detected_h = template_w, template_h
        else:
            detected_w, detected_h = template_w, template_h
        
        # Calculate bounding box with padding
        x1 = max(0, center_x - detected_w // 2 - self.pad)
        y1 = max(0, center_y - detected_h // 2 - self.pad)
        x2 = min(screenshot.shape[1], center_x + detected_w // 2 + self.pad)
        y2 = min(screenshot.shape[0], center_y + detected_h // 2 + self.pad)
        
        # Crop the region
        cropped = screenshot[y1:y2, x1:x2]
        
        if cropped.size == 0:
            logger.warning("Cropped region is empty")
            return {
                'position': (center_x, center_y),
                'confidence': match_confidence,
                'text': None,
                'text_confidence': None,
                'bbox': (x1, y1, x2, y2)
            }
        
        # Extract text using Tesseract OCR (like the sample code)
        text = None
        text_confidence = None
        
        reader = self._get_ocr_reader()
        if reader:
            try:
                # Preprocess for better OCR (like the sample code)
                gray_crop = _bgr_to_gray_cv2(cropped)
                # Apply thresholding with fallback
                if hasattr(cv2, 'threshold') and hasattr(cv2, 'THRESH_BINARY') and hasattr(cv2, 'THRESH_OTSU'):
                    gray_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                else:
                    # Manual thresholding fallback
                    from ..utils.tesseract_ocr import _threshold_otsu
                    gray_crop = _threshold_otsu(gray_crop)
                
                # Extract text using Tesseract
                results = reader.readtext(gray_crop, detail=1, paragraph=False)
                
                if results:
                    texts = []
                    confidences = []
                    for (bbox, txt, conf) in results:
                        texts.append(txt)
                        confidences.append(conf)
                    
                    text = " ".join(texts).strip()
                    text_confidence = float(np.mean(confidences)) if confidences else None
                    
                    logger.debug(f"Extracted text from {template_name}: '{text}' (confidence: {text_confidence:.3f})")
            except Exception as e:
                logger.warning(f"Error extracting text: {e}")
        
        return {
            'position': (center_x, center_y),
            'confidence': match_confidence,
            'text': text,
            'text_confidence': text_confidence,
            'bbox': (x1, y1, x2, y2),
            'cropped_region': cropped
        }
    
    def extract_text_from_region(self, image: np.ndarray, 
                                 bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        """
        Extract text from a specific region in an image
        
        Args:
            image: Image to extract text from
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary with 'text' and 'text_confidence', or None if OCR not available
        """
        x1, y1, x2, y2 = bbox
        
        # Crop the region
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return None
        
        reader = self._get_ocr_reader()
        if not reader:
            return None
        
        try:
            # Preprocess for better OCR (like the sample code)
            gray = _bgr_to_gray_cv2(cropped)
            # Apply thresholding with fallback
            if hasattr(cv2, 'threshold') and hasattr(cv2, 'THRESH_BINARY') and hasattr(cv2, 'THRESH_OTSU'):
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            else:
                # Manual thresholding fallback
                from ..utils.tesseract_ocr import _threshold_otsu
                gray = _threshold_otsu(gray)
            
            # Extract text using Tesseract
            results = reader.readtext(gray, detail=1, paragraph=False)
            
            if results:
                texts = []
                confidences = []
                for (bbox, txt, conf) in results:
                    texts.append(txt)
                    confidences.append(conf)
                
                text = " ".join(texts).strip()
                text_confidence = float(np.mean(confidences))
                
                return {
                    'text': text,
                    'text_confidence': text_confidence,
                    'raw_results': results
                }
        except Exception as e:
            logger.error(f"Error extracting text from region: {e}")
        
        return None

