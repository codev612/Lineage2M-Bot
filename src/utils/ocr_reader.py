"""
Shared EasyOCR reader singleton to avoid multiple instances consuming memory
Each EasyOCR reader can consume 1-2GB of memory, so we share a single instance
"""

from typing import Optional
import threading
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Suppress PyTorch pin_memory warnings when no GPU is available
warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)

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
    
    def _check_gpu_available(self) -> bool:
        """
        Check if GPU (CUDA) is available for EasyOCR
        
        Returns:
            True if GPU is available, False otherwise
        """
        try:
            import torch
            # Check if CUDA is available in PyTorch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    cuda_version = torch.version.cuda
                    logger.info(f"GPU detected: {device_name} (CUDA {cuda_version} available)")
                    
                    # Verify GPU can actually be used (not just detected)
                    try:
                        # Try to create a small tensor on GPU to verify it works
                        test_tensor = torch.zeros(1).cuda()
                        del test_tensor
                        torch.cuda.empty_cache()
                        logger.info(f"GPU verification successful - GPU will be used for EasyOCR")
                        return True
                    except Exception as gpu_error:
                        logger.warning(f"GPU detected but cannot be used: {gpu_error}, falling back to CPU")
                        return False
                else:
                    logger.debug("CUDA available but no GPU devices found")
                    return False
            else:
                logger.warning("CUDA not available - EasyOCR will use CPU (this will use ~1-2GB system RAM)")
                logger.warning("To use GPU, ensure PyTorch with CUDA support is installed")
                return False
        except ImportError:
            logger.warning("PyTorch not available - cannot check GPU status, EasyOCR will use CPU")
            return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}, falling back to CPU")
            return False
    
    def get_reader(self):
        """
        Get or create the shared EasyOCR reader (lazy initialization)
        Returns None if EasyOCR is not available
        Only initializes when actually needed, not at import time
        """
        if self._initialized:
            return self._reader
        
        with self._lock:
            if self._initialized:
                return self._reader
            
            # Check if EasyOCR is available first
            try:
                import easyocr
            except ImportError:
                logger.debug("EasyOCR not available")
                self._initialized = True  # Mark as initialized so we don't try again
                return None
            
            # Only initialize if we actually need OCR (lazy loading)
            try:
                # Check if GPU is available
                use_gpu = self._check_gpu_available()
                
                if use_gpu:
                    logger.info("Initializing shared EasyOCR reader with GPU acceleration...")
                    logger.info("⚠️  NOTE: If GPU initialization fails, check that PyTorch CUDA is properly installed")
                else:
                    logger.warning("⚠️  Initializing EasyOCR with CPU - this will use ~1-2GB system RAM")
                    logger.warning("⚠️  To use GPU and reduce system RAM usage, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                
                # Suppress warnings during initialization
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    # Try to initialize with GPU if available
                    try:
                        if use_gpu:
                            logger.info("Attempting GPU initialization...")
                            # Explicitly set PyTorch to use GPU
                            import torch
                            torch.cuda.set_device(0)  # Use first GPU
                            logger.info("PyTorch device set to CUDA:0")
                        
                        # Force GPU usage by setting device explicitly
                        if use_gpu:
                            import torch
                            # Ensure CUDA is available and set as default device
                            if not torch.cuda.is_available():
                                logger.error("CUDA not available despite check passing!")
                                raise RuntimeError("CUDA not available")
                            
                            # Use newer PyTorch API to set default device
                            try:
                                torch.set_default_device('cuda')
                                logger.info("Set PyTorch default device to CUDA")
                            except AttributeError:
                                # Fallback for older PyTorch versions
                                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                                logger.info("Set PyTorch default tensor type to CUDA (legacy method)")
                        
                        self._reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                        
                        # After initialization, try to move models to GPU explicitly if possible
                        if use_gpu and hasattr(self._reader, 'detector'):
                            try:
                                import torch
                                if hasattr(self._reader.detector, 'model'):
                                    if hasattr(self._reader.detector.model, 'to'):
                                        self._reader.detector.model = self._reader.detector.model.to('cuda')
                                        logger.info("Moved detector model to GPU explicitly")
                                if hasattr(self._reader, 'recognizer') and hasattr(self._reader.recognizer, 'model'):
                                    if hasattr(self._reader.recognizer.model, 'to'):
                                        self._reader.recognizer.model = self._reader.recognizer.model.to('cuda')
                                        logger.info("Moved recognizer model to GPU explicitly")
                            except Exception as move_error:
                                logger.debug(f"Could not explicitly move models to GPU: {move_error}")
                        
                        if use_gpu:
                            # Force model loading by doing a test OCR operation
                            # This ensures models are loaded to GPU memory
                            logger.info("Forcing model loading to GPU with test OCR...")
                            try:
                                import numpy as np
                                import torch
                                
                                # Create a small test image
                                test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
                                # Perform OCR to force model loading
                                _ = self._reader.readtext(test_image)
                                
                                # Clear GPU cache and check memory
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                
                                # Verify GPU is actually being used
                                if torch.cuda.is_available():
                                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
                                    
                                    if gpu_memory_allocated > 0.5:  # At least 500MB (models should be ~1-2GB)
                                        logger.info(f"✅ Shared EasyOCR reader initialized successfully with GPU acceleration")
                                        logger.info(f"✅ GPU Memory: {gpu_memory_allocated:.2f} GB allocated, {gpu_memory_reserved:.2f} GB reserved")
                                        logger.info(f"✅ Model loaded in GPU VRAM - NOT using system RAM for model")
                                    elif gpu_memory_allocated > 0.1:
                                        logger.warning(f"⚠️  GPU initialized but model may be partially in system RAM")
                                        logger.warning(f"⚠️  GPU Memory: {gpu_memory_allocated:.2f} GB (expected ~1-2GB for full model)")
                                        logger.warning(f"⚠️  Some model components may still be using system RAM")
                                    else:
                                        logger.error(f"❌ GPU initialized but very little GPU memory used ({gpu_memory_allocated:.2f} GB)")
                                        logger.error(f"❌ Models are likely loaded in system RAM instead of GPU VRAM")
                                        logger.error(f"❌ This will use ~1-2GB system RAM. Check EasyOCR/PyTorch installation.")
                                else:
                                    logger.warning("⚠️  GPU was requested but CUDA not available after initialization")
                            except Exception as load_error:
                                logger.warning(f"Could not force model loading: {load_error}")
                                # Try to check memory anyway
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                                        logger.info(f"GPU Memory after init: {gpu_memory_allocated:.2f} GB")
                                except:
                                    pass
                        else:
                            logger.warning("⚠️  Shared EasyOCR reader initialized in CPU mode")
                            logger.warning("⚠️  This uses ~1-2GB system RAM for the model")
                            logger.warning("⚠️  Install PyTorch with CUDA to use GPU and reduce system RAM usage")
                    except Exception as gpu_error:
                        # If GPU initialization fails, fallback to CPU
                        if use_gpu:
                            logger.error(f"❌ GPU initialization failed: {gpu_error}")
                            logger.warning("⚠️  Falling back to CPU mode (will use ~1-2GB system RAM)")
                            logger.warning("⚠️  To fix: Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                            try:
                                self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                                logger.info("Shared EasyOCR reader initialized successfully (CPU fallback)")
                            except Exception as cpu_error:
                                logger.error(f"CPU initialization also failed: {cpu_error}")
                                raise
                        else:
                            # Re-raise if CPU also fails
                            raise
                
                self._initialized = True
                return self._reader
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

