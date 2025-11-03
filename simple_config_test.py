#!/usr/bin/env python3
"""
Simple config test
"""

import yaml
from pathlib import Path

config_file = Path("config/bot_config.yaml")

print("🧪 Simple Config Test")
print("=" * 30)

# Read config directly
with open(config_file, 'r') as f:
    config_data = yaml.safe_load(f)

game_packages = config_data['game']['packages']
print(f"📦 Packages in config: {len(game_packages)}")
for i, pkg in enumerate(game_packages, 1):
    print(f"  {i}. {pkg}")

print(f"\n✅ Found 'com.ncsoft.lineage2mnu': {'com.ncsoft.lineage2mnu' in game_packages}")