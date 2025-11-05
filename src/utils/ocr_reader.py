"""
Shared EasyOCR reader singleton to avoid multiple instances consuming memory
Each EasyOCR reader can consume 1-2GB of memory, so we share a single instance
"""

from typing import Optional
import threading

from ..utils.logger import get_logger

logger = get_logger(__name__)

class SharedOCRReader:
    """
    Singleton wrapper for EasyOCR reader to avoid multiple instances
    """
    _instance: Optional[object] = None
    _lock = threading.Lock()
    _reader = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SharedOCRReader, cls).__new__(cls)
        return cls._instance
    
    def get_reader(self):
        """
        Get or create the shared EasyOCR reader
        Returns None if EasyOCR is not available
        """
        if self._initialized:
            return self._reader
        
        with self._lock:
            if self._initialized:
                return self._reader
            
            try:
                import easyocr
                logger.info("Initializing shared EasyOCR reader (this may take a moment and use ~1-2GB memory)...")
                self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self._initialized = True
                logger.info("Shared EasyOCR reader initialized successfully")
                return self._reader
            except ImportError:
                logger.debug("EasyOCR not available")
                self._initialized = True  # Mark as initialized so we don't try again
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR reader: {e}")
                self._initialized = True  # Mark as initialized so we don't try again
                return None
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available and initialized"""
        if not self._initialized:
            self.get_reader()
        return self._reader is not None

# Global singleton instance
shared_ocr_reader = SharedOCRReader()

