#!/usr/bin/env python3
"""
Detailed trace of configuration loading process
"""

import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataclasses import dataclass, asdict

@dataclass
class GameConfig:
    """Game-specific configuration"""
    packages: list = None
    detection_interval: float = 5.0
    auto_launch: bool = False
    
    def __post_init__(self):
        print(f"🔍 GameConfig.__post_init__ called with packages: {self.packages}")
        if self.packages is None:
            print("   ⚠️  packages is None, using defaults")
            self.packages = [
                'com.ncsoft.lineage2m',
                'com.ncsoft.lineage2m.android',
                'com.ncsoft.lineage2m.global',
                'com.ncsoft.lineage2m.sea',
                'com.ncsoft.lineage2m.kr'
            ]
        else:
            print(f"   ✅ packages already set: {self.packages}")

def main():
    print("🔧 Tracing configuration loading...")
    
    # 1. Load YAML file directly
    config_file = "config/bot_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        file_config = yaml.safe_load(f)
    
    print(f"📄 Raw YAML game section: {file_config.get('game', {})}")
    
    # 2. Test creating GameConfig with YAML data
    game_data = file_config.get('game', {})
    print(f"\n🎮 Creating GameConfig with: {game_data}")
    
    game_config = GameConfig(**game_data)
    print(f"📦 Final packages: {game_config.packages}")
    
    # 3. Test with empty dict (simulating None case)
    print(f"\n🧪 Testing with empty dict...")
    empty_game_config = GameConfig(**{})
    print(f"📦 Empty config packages: {empty_game_config.packages}")
    
    # 4. Test with explicit None
    print(f"\n🧪 Testing with explicit None...")
    none_game_config = GameConfig(packages=None)
    print(f"📦 None config packages: {none_game_config.packages}")

if __name__ == "__main__":
    main()