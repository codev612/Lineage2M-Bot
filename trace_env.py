#!/usr/bin/env python3
"""
Trace environment variable overrides in ConfigManager
"""

import os
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import ConfigManager

# Monkey patch to trace environment overrides
original_apply_env = ConfigManager._apply_environment_overrides

def debug_apply_env(self, config_data):
    print("🌍 ConfigManager._apply_environment_overrides() called")
    print(f"   📄 Input config_data game packages: {config_data.get('game', {}).get('packages', 'NOT_SET')}")
    
    # Check environment variables
    game_packages_env = os.getenv('LINEAGE2M_GAME_PACKAGES')
    print(f"   🔍 LINEAGE2M_GAME_PACKAGES env var: {game_packages_env}")
    
    result = original_apply_env(self, config_data)
    print(f"   ✅ Output config_data game packages: {result.get('game', {}).get('packages', 'NOT_SET')}")
    return result

ConfigManager._apply_environment_overrides = debug_apply_env

def main():
    print("🔧 Tracing environment overrides...")
    
    # Create a fresh ConfigManager
    cm = ConfigManager()
    
    print(f"\n📦 Final game packages: {cm.get_game_config().packages}")

if __name__ == "__main__":
    main()