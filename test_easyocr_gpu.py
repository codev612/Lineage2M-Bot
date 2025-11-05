#!/usr/bin/env python3
"""
Test script to verify EasyOCR is using GPU
"""

import sys
import torch
import numpy as np

print("=" * 60)
print("EasyOCR GPU Usage Test")
print("=" * 60)

print(f"\n1. PyTorch Info:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print(f"\n2. Initial GPU Memory:")
if torch.cuda.is_available():
    initial_allocated = torch.cuda.memory_allocated(0) / 1024**3
    initial_reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   Allocated: {initial_allocated:.3f} GB")
    print(f"   Reserved: {initial_reserved:.3f} GB")

print(f"\n3. Initializing EasyOCR with GPU=True...")
try:
    import easyocr
    print(f"   EasyOCR version: {easyocr.__version__}")
    
    # Set default tensor type to CUDA to force GPU usage
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f"   Set PyTorch default tensor type to CUDA")
    
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print(f"   EasyOCR Reader initialized")
    
    print(f"\n4. GPU Memory after EasyOCR init (before test OCR):")
    if torch.cuda.is_available():
        after_init_allocated = torch.cuda.memory_allocated(0) / 1024**3
        after_init_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   Allocated: {after_init_allocated:.3f} GB")
        print(f"   Reserved: {after_init_reserved:.3f} GB")
        print(f"   Increase: {after_init_allocated - initial_allocated:.3f} GB")
    
    print(f"\n5. Performing test OCR to force model loading...")
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    results = reader.readtext(test_image)
    print(f"   Test OCR completed")
    
    print(f"\n6. GPU Memory after test OCR (model should be loaded):")
    if torch.cuda.is_available():
        after_ocr_allocated = torch.cuda.memory_allocated(0) / 1024**3
        after_ocr_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   Allocated: {after_ocr_allocated:.3f} GB")
        print(f"   Reserved: {after_ocr_reserved:.3f} GB")
        print(f"   Increase from initial: {after_ocr_allocated - initial_allocated:.3f} GB")
        print(f"   Increase from after init: {after_ocr_allocated - after_init_allocated:.3f} GB")
        
        if after_ocr_allocated > 0.5:
            print(f"\n   [OK] GPU is being used! Models loaded in GPU VRAM")
        elif after_ocr_allocated > 0.1:
            print(f"\n   [WARN] Some GPU memory used but may be incomplete")
        else:
            print(f"\n   [FAIL] Very little GPU memory used - models likely in CPU RAM")
    
    # Check system RAM usage
    import psutil
    import os
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / 1024 / 1024
    print(f"\n7. System RAM Usage:")
    print(f"   Process RAM: {ram_usage_mb:.0f} MB ({ram_usage_mb/1024:.2f} GB)")
    
    if ram_usage_mb > 1500:
        print(f"   [WARN] High system RAM usage - models may be in CPU RAM")
    else:
        print(f"   [OK] System RAM usage is reasonable")
    
    del reader
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

