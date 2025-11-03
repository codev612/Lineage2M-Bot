"""
Configuration management for Lineage 2M Bot
Handles loading and managing configuration from YAML files and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from ..utils.logger import get_logger
from ..utils.exceptions import ConfigurationError

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class ADBConfig:
    """ADB configuration settings"""
    timeout: int = 30
    screenshot_timeout: int = 10
    connection_retry_count: int = 3
    connection_retry_delay: float = 2.0

@dataclass
class BlueStacksConfig:
    """BlueStacks emulator configuration"""
    ports: list = None
    connection_timeout: int = 15
    auto_discover: bool = True
    
    def __post_init__(self):
        if self.ports is None:
            self.ports = [5555, 5554, 5556, 5558, 5562, 5564, 5566, 5568]

@dataclass
class GameConfig:
    """Game-specific configuration"""
    packages: list = None
    detection_interval: float = 5.0
    auto_launch: bool = False
    
    def __post_init__(self):
        if self.packages is None:
            self.packages = [
                'com.ncsoft.lineage2m',
                'com.ncsoft.lineage2m.android',
                'com.ncsoft.lineage2m.global',
                'com.ncsoft.lineage2m.sea',
                'com.ncsoft.lineage2m.kr'
            ]

@dataclass
class TouchConfig:
    """Touch automation configuration"""
    tap_duration: float = 0.1
    swipe_duration: float = 0.5
    drag_duration: float = 1.0
    coordinate_scaling: bool = True
    validation_enabled: bool = True

@dataclass
class ImageRecognitionConfig:
    """Image recognition configuration"""
    confidence_threshold: float = 0.8
    template_matching_method: str = 'cv2.TM_CCOEFF_NORMED'
    ocr_enabled: bool = True
    ocr_language: str = 'eng'

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = 'INFO'
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    detailed_format: bool = False

@dataclass
class BotConfig:
    """Main bot configuration"""
    adb: ADBConfig
    bluestacks: BlueStacksConfig
    game: GameConfig
    touch: TouchConfig
    image_recognition: ImageRecognitionConfig
    logging: LoggingConfig
    
    # General settings
    auto_select_single_device: bool = True
    save_screenshots: bool = True
    screenshot_directory: str = "screenshots"
    
class ConfigManager:
    """
    Configuration manager for the Lineage 2M Bot
    Handles loading, validation, and access to configuration settings
    """
    
    def __init__(self, config_file: str = "config/bot_config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = Path(config_file)
        self.config: Optional[BotConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables"""
        try:
            # Load default configuration
            config_data = self._get_default_config()
            
            # Override with file configuration if exists
            if self.config_file.exists():
                logger.info(f"Loading configuration from {self.config_file}")
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    config_data = self._merge_configs(config_data, file_config)
            else:
                logger.info("No configuration file found, using defaults")
                # Create default config file
                self._create_default_config_file()
            
            # Override with environment variables
            config_data = self._apply_environment_overrides(config_data)
            
            # Create configuration objects
            self.config = BotConfig(
                adb=ADBConfig(**config_data.get('adb', {})),
                bluestacks=BlueStacksConfig(**config_data.get('bluestacks', {})),
                game=GameConfig(**config_data.get('game', {})),
                touch=TouchConfig(**config_data.get('touch', {})),
                image_recognition=ImageRecognitionConfig(**config_data.get('image_recognition', {})),
                logging=LoggingConfig(**config_data.get('logging', {})),
                **{k: v for k, v in config_data.items() if k not in [
                    'adb', 'bluestacks', 'game', 'touch', 'image_recognition', 'logging'
                ]}
            )
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'adb': {},
            'bluestacks': {},
            'game': {},
            'touch': {},
            'image_recognition': {},
            'logging': {},
            'auto_select_single_device': True,
            'save_screenshots': True,
            'screenshot_directory': 'screenshots'
        }
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_overrides(self, config_data: Dict) -> Dict:
        """Apply environment variable overrides"""
        # ADB settings
        if os.getenv('LINEAGE2M_ADB_TIMEOUT'):
            config_data['adb']['timeout'] = int(os.getenv('LINEAGE2M_ADB_TIMEOUT'))
        
        # Game settings
        if os.getenv('LINEAGE2M_GAME_PACKAGES'):
            packages = os.getenv('LINEAGE2M_GAME_PACKAGES').split(',')
            config_data['game']['packages'] = [pkg.strip() for pkg in packages]
        
        # Logging settings
        if os.getenv('LINEAGE2M_LOG_LEVEL'):
            config_data['logging']['level'] = os.getenv('LINEAGE2M_LOG_LEVEL')
        
        return config_data
    
    def _create_default_config_file(self) -> None:
        """Create a default configuration file"""
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            default_config = {
                'adb': asdict(ADBConfig()),
                'bluestacks': asdict(BlueStacksConfig()),
                'game': asdict(GameConfig()),
                'touch': asdict(TouchConfig()),
                'image_recognition': asdict(ImageRecognitionConfig()),
                'logging': asdict(LoggingConfig()),
                'auto_select_single_device': True,
                'save_screenshots': True,
                'screenshot_directory': 'screenshots'
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created default configuration file: {self.config_file}")
            
        except Exception as e:
            logger.warning(f"Could not create default config file: {e}")
    
    def get_config(self) -> BotConfig:
        """Get the current configuration"""
        if self.config is None:
            raise ConfigurationError("Configuration not loaded")
        return self.config
    
    def get_adb_config(self) -> ADBConfig:
        """Get ADB configuration"""
        return self.get_config().adb
    
    def get_game_config(self) -> GameConfig:
        """Get game configuration"""
        return self.get_config().game
    
    def get_touch_config(self) -> TouchConfig:
        """Get touch configuration"""
        return self.get_config().touch
    
    def get_image_config(self) -> ImageRecognitionConfig:
        """Get image recognition configuration"""
        return self.get_config().image_recognition
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return self.get_config().logging
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            config_data = {
                'adb': asdict(self.config.adb),
                'bluestacks': asdict(self.config.bluestacks),
                'game': asdict(self.config.game),
                'touch': asdict(self.config.touch),
                'image_recognition': asdict(self.config.image_recognition),
                'logging': asdict(self.config.logging),
                'auto_select_single_device': self.config.auto_select_single_device,
                'save_screenshots': self.config.save_screenshots,
                'screenshot_directory': self.config.screenshot_directory
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def reload_config(self) -> None:
        """Force reload configuration from file"""
        logger.info("Force reloading configuration")
        self.config = None
        self._load_config()

# Global configuration manager instance
config_manager = ConfigManager()