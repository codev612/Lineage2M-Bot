# Lineage 2M Bot - Advanced Automation Framework

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active%20Development-yellow.svg)

A comprehensive automation framework for Lineage 2M mobile game using ADB, computer vision, and advanced bot capabilities.

## Features

- ✅ ADB connection to Android emulators (BlueStacks, etc.)
- ✅ Game detection and state monitoring
- ✅ Screenshot capture and analysis
- ✅ Automated device discovery
- 🔄 Ready for automation logic expansion

## Prerequisites

1. **Android Emulator** (BlueStacks recommended)

   - Install and run BlueStacks or similar emulator
   - Enable Developer Options and USB Debugging

2. **ADB (Android Debug Bridge)**

   - Download Android SDK Platform Tools
   - Add ADB to your system PATH
   - Or download from: https://developer.android.com/studio/releases/platform-tools

3. **Python 3.7+**
   - Virtual environment is automatically configured

## Installation

1. Clone/download this project
2. The Python environment is already configured
3. Required packages are already installed:
   - adb-shell
   - opencv-python
   - numpy
   - pillow
   - psutil

## Usage

### Device Discovery

Discover all available Android devices and emulators:

```bash
D:/Projects/Lineage2M/.venv/Scripts/python.exe discover_devices.py
```

### Quick Device Connection

Connect to a specific device quickly:

```bash
# Interactive selection
D:/Projects/Lineage2M/.venv/Scripts/python.exe connect_device.py

# Direct connection
D:/Projects/Lineage2M/.venv/Scripts/python.exe connect_device.py 127.0.0.1:5555
```

### Main Bot

Start the main bot (includes device selection):

```bash
D:/Projects/Lineage2M/.venv/Scripts/python.exe bot.py
```

### Connection Test

Run the connection test to verify everything works:

```bash
D:/Projects/Lineage2M/.venv/Scripts/python.exe test_connection.py
```

### ADB Manager Only

Test just the ADB functionality:

```bash
D:/Projects/Lineage2M/.venv/Scripts/python.exe adb_manager.py
```

## Configuration

Edit `config.py` to customize:

- BlueStacks port numbers
- Lineage 2M package names
- Detection intervals
- Logging settings

## Troubleshooting

### ADB Not Found

- Download Android SDK Platform Tools
- Add the `platform-tools` folder to your system PATH
- Restart your terminal/command prompt

### No Devices Found

- Make sure your emulator is running
- Enable Developer Options in Android settings
- Enable USB Debugging
- Try connecting manually: `adb connect 127.0.0.1:5555`

### BlueStacks Connection Issues

- Check BlueStacks ADB settings
- Try different ports: 5555, 5554, 5556, 5558
- Restart BlueStacks if needed

## Project Structure

```
Lineage2M/
├── adb_manager.py      # ADB connection and device management
├── bot.py             # Main bot entry point with device selection
├── config.py          # Configuration settings
├── discover_devices.py # Device discovery utility
├── connect_device.py  # Quick device connection utility
├── test_connection.py # Connection testing utility
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Next Steps

This foundation provides:

1. ✅ ADB connection to emulator
2. ✅ Game detection
3. ✅ Screenshot capture
4. 🔄 Ready for automation logic

You can now add:

- Image recognition for UI elements
- Touch automation
- Game-specific bot logic
- Auto-farming routines
- And more!
