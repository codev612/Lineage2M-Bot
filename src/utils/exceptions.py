"""
Custom exceptions for Lineage 2M Bot
"""

class BotError(Exception):
    """Base exception for all bot-related errors"""
    pass

class ADBError(BotError):
    """Exception raised for ADB-related errors"""
    pass

class DeviceNotFoundError(BotError):
    """Exception raised when no devices are found"""
    pass

class ConnectionTimeoutError(BotError):
    """Exception raised when connection times out"""
    pass

class TouchAutomationError(BotError):
    """Exception raised for touch automation errors"""
    pass

class ImageRecognitionError(BotError):
    """Exception raised for image recognition errors"""
    pass

class GameStateError(BotError):
    """Exception raised for game state related errors"""
    pass

class ConfigurationError(BotError):
    """Exception raised for configuration errors"""
    pass