"""
Logging utilities for Lineage 2M Bot
Provides structured logging with file rotation and different log levels
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import os

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# Logger registry to avoid duplicate handlers
_loggers = {}

def get_logger(name: str, level: str = "INFO", 
               log_file: str = "lineage2m_bot.log",
               console_output: bool = True,
               detailed: bool = False) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name in logs/ directory
        console_output: Whether to output to console
        detailed: Whether to use detailed format
        
    Returns:
        Configured logger instance
    """
    
    # Return existing logger if already configured
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(DETAILED_FORMAT if detailed else DEFAULT_FORMAT)
    
    # File handler with rotation
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler with UTF-8 encoding support
    if console_output:
        # Create a custom handler that safely handles Unicode characters
        class SafeStreamHandler(logging.StreamHandler):
            """StreamHandler that safely handles Unicode encoding errors"""
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    stream.write(msg + self.terminator)
                    self.flush()
                except UnicodeEncodeError:
                    # If Unicode encoding fails, replace problematic characters
                    try:
                        msg = self.format(record)
                        # Replace Unicode characters that can't be encoded (emojis, etc.)
                        safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                        stream = self.stream
                        stream.write(safe_msg + self.terminator)
                        self.flush()
                    except Exception:
                        # If even that fails, just skip this log message
                        self.handleError(record)
                except Exception:
                    # Handle any other errors
                    self.handleError(record)
        
        try:
            # Try to use UTF-8 encoding
            import io
            if hasattr(sys.stdout, 'buffer'):
                console_handler = SafeStreamHandler(
                    io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                )
            else:
                console_handler = SafeStreamHandler(sys.stdout)
        except (AttributeError, io.UnsupportedOperation, Exception):
            # Fallback to safe handler with system stdout
            console_handler = SafeStreamHandler(sys.stdout)
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _loggers[name] = logger
    return logger

def setup_module_logger(module_name: str) -> logging.Logger:
    """
    Setup a logger for a specific module
    
    Args:
        module_name: Name of the module
        
    Returns:
        Configured logger
    """
    return get_logger(
        module_name,
        level="INFO",
        log_file=f"{module_name.replace('.', '_')}.log",
        console_output=True
    )

def get_debug_logger(name: str) -> logging.Logger:
    """Get a debug-level logger"""
    return get_logger(name, level="DEBUG", detailed=True)

def get_error_logger(name: str) -> logging.Logger:
    """Get an error-level logger"""
    return get_logger(name, level="ERROR", log_file="errors.log")

# Create main bot logger
main_logger = get_logger("lineage2m_bot", level="INFO", log_file="main.log")

# Create separate loggers for different components
adb_logger = get_logger("adb_manager", level="INFO", log_file="adb.log")
touch_logger = get_logger("touch_automation", level="INFO", log_file="touch.log")
vision_logger = get_logger("image_recognition", level="INFO", log_file="vision.log")
game_logger = get_logger("game_detector", level="INFO", log_file="game.log")