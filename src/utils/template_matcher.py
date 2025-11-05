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
        method_str = self.config.image_recognition.template_matching_method
        self.matching_method = getattr(cv2, method_str.split('.')[-1], cv2.TM_CCOEFF_NORMED)
        
        # Cache for loaded templates
        self.template_cache: Dict[str, np.ndarray] = {}
        
        # Multi-scale matching parameters
        self.scales = np.linspace(0.5, 2.0, 20)  # Default scales to try
        self.pad = 12  # Padding around detected region for OCR
        
        # OCR reader (lazy initialization - using Tesseract instead of EasyOCR)
        self._ocr_reader = None
        
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
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
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
    
    def _find_template_singlescale(self, search_area: np.ndarray, template: np.ndarray,
                                   threshold: float, region: Tuple[int, int, int, int],
                                   template_name: str) -> Optional[Tuple[int, int, float]]:
        """Single-scale template matching"""
        # Convert to grayscale for matching
        if len(search_area.shape) == 3:
            search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = search_area
            
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        
        # Perform template matching
        result = cv2.matchTemplate(search_gray, template_gray, self.matching_method)
        
        # Get the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For TM_CCOEFF_NORMED, higher is better
        if self.matching_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            confidence_score = 1 - min_val
            match_loc = min_loc
        else:
            confidence_score = max_val
            match_loc = max_loc
        
        # Check if confidence meets threshold
        if confidence_score < threshold:
            logger.debug(f"Template {template_name} found but confidence {confidence_score:.2f} < {threshold}")
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
            search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = search_area
            
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        
        best_val = -1
        best_loc = None
        best_size = None
        
        logger.debug(f"Multi-scale matching: trying {len(self.scales)} scales")
        
        for scale in self.scales:
            # Resize template
            resized = cv2.resize(
                template_gray,
                (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            
            # Skip if resized template is larger than search area
            if resized.shape[0] > search_gray.shape[0] or resized.shape[1] > search_gray.shape[1]:
                continue
            
            # Match template
            res = cv2.matchTemplate(search_gray, resized, self.matching_method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_size = resized.shape[::-1]  # (width, height)
        
        # Check if confidence meets threshold
        if best_val < threshold or best_loc is None:
            logger.debug(f"Multi-scale match confidence {best_val:.3f} < {threshold}")
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
            result = cv2.matchTemplate(screenshot, template, self.matching_method)
            
            # Find all matches above threshold
            if self.matching_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
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
                gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                gray_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
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
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
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

