"""
Lineage 2M Bot - Main Package
A comprehensive automation framework for Lineage 2M mobile game.
"""

__version__ = "1.0.0"
__author__ = "Lineage2M Bot Team"
__description__ = "Advanced automation framework for Lineage 2M using ADB and computer vision"

# Core imports (import only what exists)
try:
    from .core.adb_manager import ADBManager
    from .core.device_manager import DeviceManager
    # from .core.bot_core import BotCore  # Will add later
except ImportError:
    pass

# Module imports (import only what exists)  
try:
    from .modules.game_detector import GameDetector
    # from .modules.touch_automation import TouchAutomation  # Will add later
    # from .modules.screen_capture import ScreenCapture  # Will add later
    # from .modules.image_recognition import ImageRecognition  # Will add later
except ImportError:
    pass

__all__ = [
    'ADBManager',
    'DeviceManager', 
    'GameDetector'
]