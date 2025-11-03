"""
Main entry point for the Lineage 2M Bot
Provides command-line interface and bot initialization
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.device_manager import DeviceManager
from src.modules.game_detector import GameDetector
from src.utils.config import config_manager
from src.utils.logger import get_logger
from src.utils.exceptions import BotError

logger = get_logger(__name__)

class Lineage2MBot:
    """Main bot class that orchestrates all components"""
    
    def __init__(self):
        """Initialize the bot"""
        self.config = config_manager.get_config()
        self.device_manager = DeviceManager()
        self.game_detector = None
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize the bot - discover devices and connect"""
        logger.info("Initializing Lineage 2M Bot...")
        
        try:
            # Discover and select device
            if not self._setup_device():
                return False
            
            # Initialize game detector
            self.game_detector = GameDetector(self.device_manager.adb, self.config.game)
            
            # Display connection info
            self._display_connection_info()
            
            logger.info("✅ Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            return False
    
    def _setup_device(self) -> bool:
        """Setup device connection"""
        print("\n" + "="*60)
        print("📱 DEVICE DISCOVERY")
        print("="*60)
        
        # Discover devices with game priority
        devices = self.device_manager.discover_devices_with_game_priority()
        
        if not devices:
            print("❌ No devices found!")
            print("\n💡 Make sure:")
            print("   • BlueStacks or other emulator is running")
            print("   • USB debugging is enabled")
            print("   • Try: adb connect 127.0.0.1:5555")
            return False
        
        # Select device
        if self.config.auto_select_single_device and len(devices) == 1:
            self.device_manager.selected_device = devices[0]
            print(f"🎯 Auto-selected: {devices[0]['id']}")
        else:
            if not self.device_manager.select_device_interactive():
                return False
        
        # Connect to selected device
        if not self.device_manager.connect_to_selected_device():
            return False
        
        return True
    
    def _display_connection_info(self):
        """Display connection information"""
        device_info = self.device_manager.get_selected_device_info()
        
        print("\n" + "="*60)
        print("🔗 CONNECTION ESTABLISHED")
        print("="*60)
        print(f"Device ID: {device_info['id']}")
        print(f"Type: {device_info['type']}")
        print(f"Model: {device_info['model']}")
        print(f"Android: {device_info['android_version']}")
        print(f"Resolution: {device_info['resolution']}")
        print("="*60)
    
    def run_monitoring(self):
        """Run continuous monitoring of game state"""
        logger.info("Starting game monitoring...")
        self.running = True
        
        try:
            while self.running:
                self._check_game_status()
                
                import time
                time.sleep(self.config.game.detection_interval)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.running = False
    
    def _check_game_status(self):
        """Check if Lineage 2M is running and get game state"""
        is_running, package_name = self.game_detector.is_lineage2m_running()
        
        if is_running:
            logger.info(f"✅ Lineage 2M is running (Package: {package_name})")
            
            # Get game state
            game_state = self.game_detector.detect_game_state()
            logger.info(f"Game state: {game_state}")
            
            return True, package_name, game_state
        else:
            foreground_app = self.device_manager.adb.get_foreground_app()
            logger.warning(f"❌ Lineage 2M not detected. Current app: {foreground_app}")
            return False, None, None
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Bot stopping...")
        
        if self.device_manager:
            self.device_manager.disconnect()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Lineage 2M Bot - Advanced automation framework")
    parser.add_argument('--device', '-d', help='Device ID to connect to')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--discover-only', action='store_true', help='Only discover devices and exit')
    parser.add_argument('--version', action='version', version='Lineage 2M Bot v1.0.0')
    
    args = parser.parse_args()
    
    try:
        print("=" * 50)
        print("🎮 Lineage 2M Bot Starting...")
        print("=" * 50)
        
        # Initialize bot
        bot = Lineage2MBot()
        
        if args.discover_only:
            # Just discover devices and exit
            devices = bot.device_manager.discover_devices_with_game_priority()
            bot.device_manager.display_devices()
            return 0
        
        # Connect to specific device if provided
        if args.device:
            if not bot.device_manager.select_device_by_id(args.device):
                print(f"❌ Device {args.device} not found!")
                return 1
            if not bot.device_manager.connect_to_selected_device():
                print(f"❌ Failed to connect to {args.device}")
                return 1
            bot._display_connection_info()
        else:
            # Normal initialization
            if not bot.initialize():
                print("❌ Bot initialization failed!")
                return 1
        
        print("\n✅ Bot initialized successfully!")
        print("\nPress Ctrl+C to stop the bot\n")
        
        # Start monitoring
        bot.run_monitoring()
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except BotError as e:
        print(f"❌ Bot error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Unexpected error occurred")
        return 1
    finally:
        print("\n🛑 Bot stopped.")
    
    return 0

if __name__ == "__main__":
    exit(main())