# Lineage 2M Bot Configuration

# ADB Settings
ADB_TIMEOUT = 30
SCREENSHOT_TIMEOUT = 10

# BlueStacks Default Ports
BLUESTACKS_PORTS = [5555, 5554, 5556, 5558, 5562, 5564]

# Lineage 2M Package Names (vary by region)
LINEAGE2M_PACKAGES = [
    'com.ncsoft.lineage2m',
    'com.ncsoft.lineage2m.android', 
    'com.ncsoft.lineage2m.global',
    'com.ncsoft.lineage2m.sea',
    'com.ncsoft.lineage2m.kr'
]

# Game Detection Settings
DETECTION_INTERVAL = 5  # seconds between game state checks
MAX_RETRIES = 3

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'lineage2m_bot.log'