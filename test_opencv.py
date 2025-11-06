"""
Test OpenCV installation and functionality
This script checks if OpenCV is properly installed and all required functions are available
"""

import sys
import subprocess
import os

def test_opencv_installation():
    """Test if OpenCV is properly installed and functional"""
    print("=" * 60)
    print("Testing OpenCV Installation")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # Test 1: Import OpenCV
    print("\n1. Testing OpenCV import...")
    try:
        import cv2
        print(f"   [OK] OpenCV imported successfully")
        try:
            version = cv2.__version__
            print(f"   Version: {version}")
        except AttributeError:
            print(f"   [WARNING] Version information not available (installation may be corrupted)")
            warnings.append("OpenCV version attribute not available")
    except ImportError as e:
        print(f"   [FAIL] Failed to import OpenCV: {e}")
        issues.append("OpenCV cannot be imported")
        return False, issues, warnings
    
    # Test 2: Check core functions
    print("\n2. Testing core OpenCV functions...")
    core_functions = [
        'imread', 'imwrite', 'cvtColor', 'resize', 
        'matchTemplate', 'minMaxLoc', 'Canny',
        'rectangle', 'putText', 'findContours',
        'contourArea', 'threshold'
    ]
    
    missing_functions = []
    for func in core_functions:
        if hasattr(cv2, func):
            print(f"   [OK] cv2.{func} is available")
        else:
            print(f"   [FAIL] cv2.{func} is MISSING")
            missing_functions.append(func)
            issues.append(f"cv2.{func} is not available")
    
    # Test 3: Check constants
    print("\n3. Testing OpenCV constants...")
    constants = [
        'COLOR_BGR2GRAY', 'COLOR_BGR2HSV', 'COLOR_RGB2BGR',
        'IMREAD_COLOR', 'IMREAD_UNCHANGED',
        'FONT_HERSHEY_SIMPLEX',
        'TM_CCOEFF_NORMED', 'TM_SQDIFF_NORMED',
        'THRESH_BINARY', 'THRESH_OTSU',
        'INTER_CUBIC', 'INTER_LINEAR',
        'RETR_EXTERNAL', 'CHAIN_APPROX_SIMPLE'
    ]
    
    missing_constants = []
    for const in constants:
        if hasattr(cv2, const):
            print(f"   [OK] cv2.{const} is available")
        else:
            print(f"   [FAIL] cv2.{const} is MISSING")
            missing_constants.append(const)
            warnings.append(f"cv2.{const} is not available (may use integer constants)")
    
    # Test 4: Test basic image operations
    print("\n4. Testing basic image operations...")
    try:
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square
        
        # Test imwrite
        if hasattr(cv2, 'imwrite'):
            test_path = 'test_opencv_image.png'
            result = cv2.imwrite(test_path, test_image)
            if result:
                print(f"   [OK] cv2.imwrite() works")
            else:
                print(f"   [FAIL] cv2.imwrite() failed")
                issues.append("cv2.imwrite() failed")
        else:
            print(f"   - cv2.imwrite() not available (skipped)")
        
        # Test imread
        if hasattr(cv2, 'imread'):
            if hasattr(cv2, 'imwrite') and os.path.exists(test_path):
                loaded = cv2.imread(test_path)
                if loaded is not None:
                    print(f"   [OK] cv2.imread() works")
                else:
                    print(f"   [FAIL] cv2.imread() failed or returned None")
                    issues.append("cv2.imread() failed")
            else:
                print(f"   - cv2.imread() test skipped (no test file available)")
        else:
            print(f"   - cv2.imread() not available (skipped)")
        
        # Test cvtColor
        if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2GRAY'):
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            if gray.shape == (100, 100):
                print(f"   [OK] cv2.cvtColor() works")
            else:
                print(f"   [FAIL] cv2.cvtColor() failed")
                issues.append("cv2.cvtColor() failed")
        else:
            print(f"   - cv2.cvtColor() not available (skipped)")
        
        # Test resize
        if hasattr(cv2, 'resize'):
            resized = cv2.resize(test_image, (50, 50))
            if resized.shape == (50, 50, 3):
                print(f"   [OK] cv2.resize() works")
            else:
                print(f"   [FAIL] cv2.resize() failed")
                issues.append("cv2.resize() failed")
        else:
            print(f"   - cv2.resize() not available (skipped)")
        
        # Test matchTemplate
        if hasattr(cv2, 'matchTemplate') and hasattr(cv2, 'minMaxLoc'):
            template = test_image[25:75, 25:75]
            if hasattr(cv2, 'TM_CCOEFF_NORMED'):
                result = cv2.matchTemplate(test_image, template, cv2.TM_CCOEFF_NORMED)
            else:
                result = cv2.matchTemplate(test_image, template, 5)  # TM_CCOEFF_NORMED = 5
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            print(f"   [OK] cv2.matchTemplate() and cv2.minMaxLoc() work")
            print(f"     Best match: {max_loc}, confidence: {max_val:.3f}")
        else:
            print(f"   - cv2.matchTemplate() or cv2.minMaxLoc() not available (skipped)")
            if 'matchTemplate' not in missing_functions:
                issues.append("cv2.matchTemplate() or cv2.minMaxLoc() not available")
        
        # Clean up test file
        if os.path.exists('test_opencv_image.png'):
            os.remove('test_opencv_image.png')
            
    except Exception as e:
        print(f"   [FAIL] Error during image operations: {e}")
        issues.append(f"Image operations failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if issues:
        print(f"\n[FAIL] Found {len(issues)} critical issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        print("\n[WARNING] OpenCV installation appears CORRUPTED or INCOMPLETE")
        print("\nRecommendation: Reinstall OpenCV")
        print("Run this command to fix:")
        print("  pip uninstall opencv-python opencv-python-headless -y")
        print("  pip install --force-reinstall --no-cache-dir opencv-python==4.8.1.78")
        return False, issues, warnings
    else:
        if warnings:
            print(f"\n[WARNING] Found {len(warnings)} warning(s) (non-critical):")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n[SUCCESS] All tests passed! OpenCV is working correctly.")
        return True, issues, warnings

def reinstall_opencv():
    """Reinstall OpenCV with proper flags"""
    print("\n" + "=" * 60)
    print("Reinstalling OpenCV...")
    print("=" * 60)
    
    try:
        # Uninstall
        print("\n1. Uninstalling existing OpenCV packages...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', 
            'opencv-python', 'opencv-python-headless', '-y'
        ], check=True)
        print("   [OK] Uninstalled")
        
        # Install
        print("\n2. Installing OpenCV...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            '--force-reinstall', '--no-cache-dir',
            'opencv-python==4.8.1.78'
        ], check=True)
        print("   [OK] Installed")
        
        print("\n[SUCCESS] OpenCV reinstalled successfully!")
        print("\nPlease run this test again to verify:")
        print("  python test_opencv.py")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Failed to reinstall OpenCV: {e}")
        return False

if __name__ == "__main__":
    print("OpenCV Installation Test")
    print("=" * 60)
    
    success, issues, warnings = test_opencv_installation()
    
    if not success:
        print("\n" + "=" * 60)
        response = input("\nWould you like to reinstall OpenCV now? (y/n): ")
        if response.lower() == 'y':
            reinstall_opencv()
        else:
            print("\nTo reinstall OpenCV manually, run:")
            print("  pip uninstall opencv-python opencv-python-headless -y")
            print("  pip install --force-reinstall --no-cache-dir opencv-python==4.8.1.78")

