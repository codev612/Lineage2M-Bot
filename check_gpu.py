#!/usr/bin/env python3
"""
Quick GPU diagnostic script to check if GPU/CUDA is available for EasyOCR
"""

import sys

print("=" * 60)
print("GPU/CUDA Diagnostic for EasyOCR")
print("=" * 60)

# Check PyTorch
print("\n1. Checking PyTorch installation...")
try:
    import torch
    print(f"   [OK] PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    print("\n2. Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"   [OK] CUDA is available!")
        print(f"   [OK] CUDA version: {torch.version.cuda}")
        print(f"   [OK] cuDNN version: {torch.backends.cudnn.version()}")
        
        # Get GPU info
        device_count = torch.cuda.device_count()
        print(f"\n3. GPU Device Information:")
        print(f"   [OK] Number of GPUs: {device_count}")
        
        for i in range(device_count):
            print(f"\n   GPU {i}:")
            print(f"   - Name: {torch.cuda.get_device_name(i)}")
            print(f"   - Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # Test GPU usage
            try:
                test_tensor = torch.zeros(1000, 1000).cuda(i)
                gpu_memory = torch.cuda.memory_allocated(i) / 1024**2
                del test_tensor
                torch.cuda.empty_cache()
                print(f"   - Test allocation: [OK] Success ({gpu_memory:.2f} MB)")
            except Exception as e:
                print(f"   - Test allocation: [FAIL] Failed - {e}")
        
        print("\n4. Testing EasyOCR GPU support...")
        try:
            import easyocr
            print("   [OK] EasyOCR is installed")
            
            # Try to initialize with GPU
            print("   Attempting to initialize EasyOCR with GPU...")
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            
            # Force model loading by doing a test read
            print("   Forcing model loading with test OCR...")
            import numpy as np
            test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White image
            try:
                reader.readtext(test_image)  # This forces model loading
            except:
                pass
            
            # Check GPU memory after initialization and model loading
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            if gpu_memory_allocated > 0.5:  # At least 500MB (model should be ~1-2GB)
                print(f"   [OK] EasyOCR initialized with GPU!")
                print(f"   [OK] GPU Memory allocated: {gpu_memory_allocated:.2f} GB")
                print(f"   [OK] GPU Memory reserved: {gpu_memory_reserved:.2f} GB")
                print(f"   [OK] Model is in GPU VRAM, NOT using system RAM!")
            elif gpu_memory_allocated > 0.1:
                print(f"   [WARN] GPU initialized but model may be partially in system RAM")
                print(f"   [WARN] GPU Memory allocated: {gpu_memory_allocated:.2f} GB")
                print(f"   [WARN] Expected ~1-2GB for full model - some may be in system RAM")
            else:
                print(f"   [WARN] GPU initialized but very little memory used ({gpu_memory_allocated:.2f} GB)")
                print(f"   [WARN] Model is likely still using system RAM (~1-2GB)")
            
            del reader
            torch.cuda.empty_cache()
            
        except ImportError:
            print("   [FAIL] EasyOCR is not installed")
            print("   Install with: pip install easyocr")
        except Exception as e:
            print(f"   [FAIL] EasyOCR GPU initialization failed: {e}")
            print(f"   This means EasyOCR will use CPU and ~1-2GB system RAM")
    else:
        print("   [FAIL] CUDA is NOT available")
        print("\n   Possible causes:")
        print("   1. PyTorch was installed without CUDA support")
        print("   2. CUDA drivers are not installed")
        print("   3. CUDA version mismatch")
        print("\n   To fix:")
        print("   - Install PyTorch with CUDA:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   - Or for CUDA 12.1:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        
except ImportError:
    print("   [FAIL] PyTorch is not installed")
    print("   Install with: pip install torch")
    print("   For GPU support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)

