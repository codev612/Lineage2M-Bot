"""
Lineage 2M Bot - Main Entry Point
"""

import logging
import time
import sys
from adb_manager import ADBManager, GameDetector
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Lineage2MBot:
    def __init__(self):
        self.adb = ADBManager()
        self.game_detector = None
        self.running = False
        self.selected_device = None
        
    def discover_and_list_devices(self) -> bool:
        """Discover all available devices and let user choose"""
        logger.info("🔍 Discovering available devices...")
        print("\n" + "="*60)
        print("📱 DEVICE DISCOVERY")
        print("="*60)
        
        # Check if ADB is available first
        if not self.adb.check_adb_available():
            logger.error("❌ ADB not found! Please install Android SDK Platform Tools.")
            return False
        
        # Get all available devices
        devices = self.adb.get_all_available_devices()
        
        if not devices:
            print("❌ No devices found!")
            print("\n💡 Make sure:")
            print("   • BlueStacks or other emulator is running")
            print("   • USB debugging is enabled")
            print("   • Try: adb connect 127.0.0.1:5555")
            return False
        
        # Display devices
        print(f"\n✅ Found {len(devices)} device(s):")
        print("-" * 60)
        
        for i, device in enumerate(devices, 1):
            status_icon = "🟢" if device['status'] == 'connected' else "🟡"
            print(f"{i}. {status_icon} {device['id']}")
            print(f"   Type: {device['type']}")
            print(f"   Model: {device['model']}")
            print(f"   Android: {device['android_version']} (API {device['api_level']})")
            print(f"   Resolution: {device['resolution']}")
            print(f"   Status: {device['status']}")
            print("-" * 60)
        
        # Auto-select if only one device
        if len(devices) == 1:
            self.selected_device = devices[0]
            print(f"🎯 Auto-selected: {self.selected_device['id']}")
            return True
        
        # Let user choose
        while True:
            try:
                choice = input(f"\n👆 Select device (1-{len(devices)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return False
                
                device_index = int(choice) - 1
                if 0 <= device_index < len(devices):
                    self.selected_device = devices[device_index]
                    print(f"✅ Selected: {self.selected_device['id']}")
                    return True
                else:
                    print(f"❌ Please enter a number between 1 and {len(devices)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return False
    
    def initialize(self) -> bool:
        """Initialize the bot - discover devices and connect"""
        logger.info("Initializing Lineage 2M Bot...")
        
        # First, discover and select device
        if not self.discover_and_list_devices():
            return False
        
        # Connect to selected device
        device_id = self.selected_device['id']
        
        if self.selected_device['status'] == 'available':
            # Need to connect first
            logger.info(f"Connecting to {device_id}...")
            if not self.adb.connect_to_device(device_id):
                logger.error(f"Failed to connect to {device_id}")
                return False
        else:
            # Already connected, just set it
            self.adb.device_id = device_id
            self.adb.connected = True
        
        # Initialize game detector
        self.game_detector = GameDetector(self.adb)
        
        # Display final connection info
        print("\n" + "="*60)
        print("🔗 CONNECTION ESTABLISHED")
        print("="*60)
        print(f"Device ID: {self.selected_device['id']}")
        print(f"Type: {self.selected_device['type']}")
        print(f"Model: {self.selected_device['model']}")
        print(f"Android: {self.selected_device['android_version']}")
        print(f"Resolution: {self.selected_device['resolution']}")
        print("="*60)
        
        return True
    
    def check_game_status(self):
        """Check if Lineage 2M is running and get game state"""
        is_running, package_name = self.game_detector.is_lineage2m_running()
        
        if is_running:
            logger.info(f"✅ Lineage 2M is running (Package: {package_name})")
            
            # Get game state
            game_state = self.game_detector.detect_game_state()
            logger.info(f"Game state: {game_state}")
            
            return True, package_name, game_state
        else:
            foreground_app = self.adb.get_foreground_app()
            logger.warning(f"❌ Lineage 2M not detected. Current app: {foreground_app}")
            return False, None, None
    
    def run_monitoring(self):
        """Run continuous monitoring of game state"""
        logger.info("Starting game monitoring...")
        self.running = True
        
        try:
            while self.running:
                is_running, package, state = self.check_game_status()
                
                if is_running:
                    logger.info("Game is active - ready for automation")
                    # Here you would add your bot logic
                else:
                    logger.info("Waiting for game to start...")
                
                time.sleep(config.DETECTION_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Bot stopping...")

def main():
    """Main function"""
    print("=" * 50)
    print("🎮 Lineage 2M Bot Starting...")
    print("=" * 50)
    
    bot = Lineage2MBot()
    
    if not bot.initialize():
        print("❌ Bot initialization failed!")
        return 1
    
    print("✅ Bot initialized successfully!")
    print("\nPress Ctrl+C to stop the bot\n")
    
    try:
        bot.run_monitoring()
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop()
    
    print("\n🛑 Bot stopped.")
    return 0

if __name__ == "__main__":
    exit(main())