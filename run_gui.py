#!/usr/bin/env python3
"""
GUI Launcher for Lineage 2M Bot
Main entry point for the desktop application
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import the GUI
from src.gui.main_window import MainWindow
from src.utils.logger import get_logger
from src.utils.config import config_manager

logger = get_logger(__name__)

def main():
    """Main entry point for the GUI application"""
    try:
        logger.info("Starting Lineage 2M Bot GUI...")
        
        # Initialize configuration
        config = config_manager.get_config()
        logger.info(f"Configuration loaded from: {config_manager.config_file}")
        
        # Create and run the GUI
        app = MainWindow()
        logger.info("GUI initialized successfully")
        
        # Start the application
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting GUI application: {e}")
        sys.exit(1)
    finally:
        logger.info("Lineage 2M Bot GUI shutdown")

if __name__ == "__main__":
    main()