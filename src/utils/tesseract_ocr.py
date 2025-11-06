"""
Tesseract OCR reader - Lightweight OCR using Tesseract instead of EasyOCR
Uses minimal memory compared to EasyOCR (~100MB vs ~2GB)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
import platform

from ..utils.logger import get_logger

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

def _threshold_otsu(gray: np.ndarray) -> np.ndarray:
    """Apply OTSU thresholding - fallback if cv2.threshold doesn't exist"""
    if hasattr(cv2, 'threshold') and hasattr(cv2, 'THRESH_BINARY') and hasattr(cv2, 'THRESH_OTSU'):
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    else:
        # Manual OTSU thresholding
        # Calculate histogram
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        hist = hist.astype(np.float32)
        
        # Calculate cumulative sums
        cumsum = np.cumsum(hist)
        cumsum_w = np.cumsum(hist * np.arange(256))
        
        # Calculate between-class variance for all thresholds
        total_mean = cumsum_w[-1] / cumsum[-1]
        between_class_var = np.zeros(256)
        
        for t in range(256):
            w0 = cumsum[t]
            if w0 == 0:
                continue
            w1 = cumsum[-1] - w0
            if w1 == 0:
                break
            
            mean0 = cumsum_w[t] / w0
            mean1 = (cumsum_w[-1] - cumsum_w[t]) / w1
            
            between_class_var[t] = w0 * w1 * (mean0 - mean1) ** 2
        
        # Find threshold with maximum variance
        threshold = np.argmax(between_class_var)
        
        # Apply threshold
        thresh = np.where(gray > threshold, 255, 0).astype(np.uint8)
        return thresh

logger = get_logger(__name__)

# Try to import pytesseract
TESSERACT_AVAILABLE = False
_pytesseract_import_error = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError as e:
    TESSERACT_AVAILABLE = False
    _pytesseract_import_error = str(e)
    # Don't log here during module import - log when actually needed
except Exception as e:
    TESSERACT_AVAILABLE = False
    _pytesseract_import_error = str(e)
    # Don't log here during module import - log when actually needed


class TesseractOCRReader:
    """
    Tesseract-based OCR reader - lightweight alternative to EasyOCR
    Uses ~100MB RAM instead of ~2GB
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, lang: str = 'eng'):
        """
        Initialize Tesseract OCR reader
        
        Args:
            tesseract_cmd: Path to tesseract executable (auto-detected if None)
            lang: Language code for OCR (default: 'eng')
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract not available. Install with: pip install pytesseract")
        
        self.lang = lang
        
        # Set Tesseract command path if provided or auto-detect
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            logger.info(f"Using Tesseract at: {tesseract_cmd}")
        else:
            # Try to auto-detect Tesseract path
            self._auto_detect_tesseract()
        
        # Verify Tesseract is actually working
        try:
            import subprocess
            test_cmd = [pytesseract.pytesseract.tesseract_cmd, '--version']
            result = subprocess.run(test_cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.decode('utf-8', errors='ignore').split('\n')[0]
                logger.info(f"Tesseract OCR reader initialized: {version} (uses ~100MB RAM vs ~2GB for EasyOCR)")
            else:
                raise RuntimeError(f"Tesseract executable returned error: {result.stderr.decode('utf-8', errors='ignore')}")
        except Exception as e:
            logger.error(f"Failed to verify Tesseract installation: {e}")
            raise RuntimeError(f"Tesseract not properly configured: {e}")
    
    def _auto_detect_tesseract(self):
        """Auto-detect Tesseract installation path"""
        system = platform.system()
        
        # Common Tesseract installation paths
        common_paths = []
        
        if system == 'Windows':
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            ]
        elif system == 'Darwin':  # macOS
            common_paths = [
                '/usr/local/bin/tesseract',
                '/opt/homebrew/bin/tesseract',
            ]
        else:  # Linux
            common_paths = [
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract',
            ]
        
        # Try to find Tesseract
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Auto-detected Tesseract at: {path}")
                return
        
        # If not found in common paths, try to use system PATH
        try:
            import subprocess
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                # Tesseract is in PATH, but we need to find the full path
                # Try common Windows paths or use 'tesseract' as-is
                if system == 'Windows':
                    # Try to find it
                    try:
                        which_result = subprocess.run(['where', 'tesseract'], 
                                                    capture_output=True, text=True, timeout=5)
                        if which_result.returncode == 0 and which_result.stdout.strip():
                            path = which_result.stdout.strip().split('\n')[0]
                            pytesseract.pytesseract.tesseract_cmd = path
                            logger.info(f"Found Tesseract in PATH: {path}")
                            return
                    except:
                        pass
                # If we can't find the full path, use 'tesseract' and hope it's in PATH
                logger.info("Tesseract found in system PATH (using 'tesseract' command)")
                return
        except Exception as e:
            logger.debug(f"Error checking system PATH for Tesseract: {e}")
        
        # If still not found, try the default path that we know exists
        default_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if system == 'Windows' and os.path.exists(default_windows_path):
            pytesseract.pytesseract.tesseract_cmd = default_windows_path
            logger.info(f"Using default Tesseract path: {default_windows_path}")
            return
        
        logger.warning("Tesseract not found in common locations. Please set tesseract_cmd manually or install Tesseract-OCR")
        raise RuntimeError("Tesseract-OCR executable not found. Please install Tesseract-OCR from https://github.com/UB-Mannheim/tesseract/wiki")
    
    def readtext(self, image: np.ndarray, detail: int = 0, paragraph: bool = False) -> List[Tuple]:
        """
        Read text from image using Tesseract (EasyOCR-compatible interface)
        
        Args:
            image: Input image (BGR or RGB numpy array)
            detail: 0 = return text only, 1 = return detailed results with bbox
            paragraph: Whether to return as paragraph (ignored for now)
            
        Returns:
            If detail=0: List of strings
            If detail=1: List of tuples (bbox, text, confidence)
                bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if not TESSERACT_AVAILABLE:
            return []
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = _bgr_to_gray(image)
            else:
                gray = image.copy()
            
            # Preprocess image for better OCR results
            # Apply thresholding (OTSU)
            thresh = _threshold_otsu(gray)
            
            if detail == 0:
                # Simple text extraction
                text = pytesseract.image_to_string(thresh, lang=self.lang)
                # Return list of lines
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                return lines
            else:
                # Detailed extraction with bounding boxes
                data = pytesseract.image_to_data(thresh, lang=self.lang, output_type=pytesseract.Output.DICT)
                
                results = []
                n_boxes = len(data['text'])
                
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i])
                    
                    # Skip empty text or low confidence
                    if not text or conf < 0:
                        continue
                    
                    # Get bounding box
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # Convert to EasyOCR bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    bbox = [
                        [x, y],           # top-left
                        [x + w, y],      # top-right
                        [x + w, y + h],  # bottom-right
                        [x, y + h]       # bottom-left
                    ]
                    
                    # Normalize confidence (Tesseract returns 0-100, EasyOCR uses 0-1)
                    confidence = conf / 100.0
                    
                    results.append((bbox, text, confidence))
                
                return results
                
        except Exception as e:
            logger.error(f"Error in Tesseract OCR: {e}")
            return []
    
    def extract_text_from_region(self, image: np.ndarray, 
                                bbox: Tuple[int, int, int, int],
                                preprocess: bool = True) -> Optional[str]:
        """
        Extract text from a specific region (like the sample code)
        
        Args:
            image: Input image (BGR)
            bbox: Bounding box (x1, y1, x2, y2)
            preprocess: Whether to apply preprocessing (thresholding)
            
        Returns:
            Extracted text string or None
        """
        if not TESSERACT_AVAILABLE:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return None
            
            # Convert to grayscale
            gray = _bgr_to_gray(cropped)
            
            if preprocess:
                # Apply thresholding (like the sample code)
                thresh = _threshold_otsu(gray)
            else:
                thresh = gray
            
            # Extract text
            text = pytesseract.image_to_string(thresh, lang=self.lang)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from region: {e}")
            return None


# Global singleton instance (similar to EasyOCR reader)
_shared_tesseract_reader: Optional[TesseractOCRReader] = None


def get_tesseract_reader(tesseract_cmd: Optional[str] = None, lang: str = 'eng') -> Optional[TesseractOCRReader]:
    """
    Get shared Tesseract OCR reader singleton
    
    Args:
        tesseract_cmd: Path to tesseract executable (only used on first call)
        lang: Language code for OCR (default: 'eng')
        
    Returns:
        TesseractOCRReader instance or None if Tesseract not available
    """
    global _shared_tesseract_reader
    
    if not TESSERACT_AVAILABLE:
        error_msg = _pytesseract_import_error or "Unknown error"
        logger.warning(f"pytesseract not available: {error_msg}")
        logger.warning("Install with: pip install pytesseract")
        logger.warning("Also ensure Tesseract-OCR executable is installed from https://github.com/UB-Mannheim/tesseract/wiki")
        return None
    
    if _shared_tesseract_reader is None:
        try:
            # If tesseract_cmd not provided, try to use the default Windows path
            if tesseract_cmd is None and platform.system() == 'Windows':
                default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.path.exists(default_path):
                    tesseract_cmd = default_path
                    logger.info(f"Using default Tesseract path: {default_path}")
            
            _shared_tesseract_reader = TesseractOCRReader(tesseract_cmd=tesseract_cmd, lang=lang)
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract OCR reader: {e}")
            logger.error(f"Please ensure Tesseract-OCR is installed from https://github.com/UB-Mannheim/tesseract/wiki")
            logger.error(f"Error details: {type(e).__name__}: {e}")
            return None
    
    return _shared_tesseract_reader

