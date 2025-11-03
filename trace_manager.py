#!/usr/bin/env python3
"""
Detailed trace of ConfigManager loading process
"""

import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import ConfigManager

# Monkey patch to add debug traces
original_load_config = ConfigManager._load_config
original_merge_configs = ConfigManager._merge_configs

def debug_load_config(self):
    print("🔧 ConfigManager._load_config() called")
    return original_load_config(self)

def debug_merge_configs(self, base, override):
    print(f"🔄 ConfigManager._merge_configs() called:")
    print(f"   📄 Base: {base}")
    print(f"   🔄 Override: {override}")
    result = original_merge_configs(self, base, override)
    print(f"   ✅ Result: {result}")
    return result

ConfigManager._load_config = debug_load_config
ConfigManager._merge_configs = debug_merge_configs

def main():
    print("🔧 Tracing ConfigManager loading...")
    
    # Create a fresh ConfigManager
    cm = ConfigManager()
    
    print(f"\n📦 Final game packages: {cm.get_game_config().packages}")
    print(f"✅ Has lineage2mnu: {'com.ncsoft.lineage2mnu' in cm.get_game_config().packages}")

if __name__ == "__main__":
    main()