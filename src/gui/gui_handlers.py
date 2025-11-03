"""
GUI Event Handlers - Methods for handling GUI events and user interactions
Extension of the MainWindow class with all event handling methods
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import json
import yaml
from pathlib import Path
import subprocess
import webbrowser
from PIL import Image, ImageTk
import cv2
import numpy as np

from ..core.device_manager import DeviceManager
from ..modules.game_detector import GameDetector
from ..utils.config import config_manager
from ..utils.logger import get_logger
from ..utils.exceptions import BotError

logger = get_logger(__name__)

class GUIEventHandlers:
    """
    Event handlers and methods for the main GUI window
    This class contains all the event handling methods for MainWindow
    """
    
    def _discover_devices(self):
        """Discover available devices"""
        def discover_thread():
            try:
                self._update_status("🔍 Discovering devices...")
                self._show_progress()
                
                # Discover devices with game priority
                devices = self.device_manager.discover_devices_with_game_priority()
                self.devices_list = devices
                
                # Update GUI in main thread
                self.message_queue.put({
                    'type': 'devices_discovered',
                    'devices': devices
                })
                
            except Exception as e:
                logger.error(f"Error discovering devices: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Device discovery failed: {e}"
                })
            finally:
                self._hide_progress()
        
        # Run in separate thread
        threading.Thread(target=discover_thread, daemon=True).start()
    
    def _refresh_devices(self):
        """Refresh the device list"""
        self._discover_devices()
    
    def _connect_to_device(self):
        """Connect to the selected device"""
        if not self.selected_device:
            messagebox.showwarning("No Device", "Please select a device first.")
            return
        
        def connect_thread():
            try:
                self._update_status("🔗 Connecting to device...")
                self._show_progress()
                
                # Connect to device
                if self.device_manager.select_device_by_id(self.selected_device['id']):
                    if self.device_manager.connect_to_selected_device():
                        # Initialize game detector
                        self.game_detector = GameDetector(
                            self.device_manager.adb, 
                            self.config.game
                        )
                        
                        self.message_queue.put({
                            'type': 'device_connected',
                            'device': self.selected_device
                        })
                    else:
                        raise BotError("Failed to connect to device")
                else:
                    raise BotError("Failed to select device")
                    
            except Exception as e:
                logger.error(f"Error connecting to device: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Connection failed: {e}"
                })
            finally:
                self._hide_progress()
        
        # Run in separate thread
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def _on_device_select(self, event):
        """Handle device selection in the tree"""
        selection = self.device_tree.selection()
        if selection:
            item_id = selection[0]
            device_id = self.device_tree.item(item_id, "text")
            
            # Find the device in our list
            for device in self.devices_list:
                if device['id'] == device_id:
                    self.selected_device = device
                    self.connect_btn.configure(state="normal")
                    break
    
    def _start_bot(self):
        """Start the bot"""
        if not self.game_detector:
            messagebox.showwarning("No Connection", "Please connect to a device first.")
            return
        
        def bot_thread():
            try:
                self.bot_running = True
                self._update_status("🤖 Bot running...")
                
                # Update GUI
                self.message_queue.put({
                    'type': 'bot_started'
                })
                
                # Main bot loop
                while self.bot_running:
                    # Check game status
                    is_running, package = self.game_detector.is_lineage2m_running()
                    
                    if is_running:
                        game_state = self.game_detector.detect_game_state()
                        
                        self.message_queue.put({
                            'type': 'game_status_update',
                            'running': True,
                            'package': package,
                            'state': game_state
                        })
                    else:
                        self.message_queue.put({
                            'type': 'game_status_update',
                            'running': False,
                            'package': None,
                            'state': None
                        })
                    
                    # Take screenshot periodically
                    if self.bot_running:
                        screenshot = self.device_manager.adb.take_screenshot()
                        if screenshot is not None:
                            self.screenshot_queue.put(screenshot)
                    
                    # Wait for configured interval
                    time.sleep(self.interval_var.get())
                
            except Exception as e:
                logger.error(f"Error in bot thread: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Bot error: {e}"
                })
            finally:
                self.bot_running = False
                self.message_queue.put({
                    'type': 'bot_stopped'
                })
        
        # Start bot thread
        threading.Thread(target=bot_thread, daemon=True).start()
    
    def _stop_bot(self):
        """Stop the bot"""
        self.bot_running = False
        self._update_status("⏹️ Stopping bot...")
    
    def _take_screenshot(self):
        """Take a manual screenshot"""
        if not self.device_manager or not self.device_manager.adb.connected:
            messagebox.showwarning("No Connection", "Please connect to a device first.")
            return
        
        def screenshot_thread():
            try:
                self._update_status("📸 Taking screenshot...")
                
                screenshot = self.device_manager.adb.take_screenshot()
                if screenshot is not None:
                    self.screenshot_queue.put(screenshot)
                    self.current_screenshot = screenshot
                    
                    self.message_queue.put({
                        'type': 'screenshot_taken'
                    })
                else:
                    raise BotError("Failed to capture screenshot")
                    
            except Exception as e:
                logger.error(f"Error taking screenshot: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Screenshot failed: {e}"
                })
        
        threading.Thread(target=screenshot_thread, daemon=True).start()
    
    def _update_interval_label(self, value):
        """Update the interval label when slider changes"""
        self.interval_label.configure(text=f"{float(value):.1f}s")
    
    def _clear_logs(self):
        """Clear the log display"""
        self.log_text.delete("1.0", tk.END)
    
    def _save_logs(self):
        """Save logs to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                log_content = self.log_text.get("1.0", tk.END)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                
                self._update_status(f"💾 Logs saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save logs: {e}")
    
    def _load_config_display(self):
        """Load configuration into the display"""
        try:
            config_dict = {
                'adb': {
                    'timeout': self.config.adb.timeout,
                    'screenshot_timeout': self.config.adb.screenshot_timeout,
                    'connection_retry_count': self.config.adb.connection_retry_count
                },
                'game': {
                    'packages': self.config.game.packages,
                    'detection_interval': self.config.game.detection_interval,
                    'auto_launch': self.config.game.auto_launch
                },
                'touch': {
                    'tap_duration': self.config.touch.tap_duration,
                    'swipe_duration': self.config.touch.swipe_duration,
                    'coordinate_scaling': self.config.touch.coordinate_scaling
                },
                'logging': {
                    'level': self.config.logging.level,
                    'file_enabled': self.config.logging.file_enabled,
                    'console_enabled': self.config.logging.console_enabled
                }
            }
            
            config_yaml = yaml.dump(config_dict, default_flow_style=False, indent=2)
            self.config_text.delete("1.0", tk.END)
            self.config_text.insert("1.0", config_yaml)
            
        except Exception as e:
            logger.error(f"Error loading config display: {e}")
    
    def _reload_config(self):
        """Reload configuration from file"""
        try:
            # Reload config manager
            config_manager._load_config()
            self.config = config_manager.get_config()
            
            # Update display
            self._load_config_display()
            
            self._update_status("🔄 Configuration reloaded")
            
        except Exception as e:
            messagebox.showerror("Config Error", f"Failed to reload config: {e}")
    
    def _save_config(self):
        """Save configuration from the display"""
        try:
            # Get config text
            config_yaml = self.config_text.get("1.0", tk.END)
            config_dict = yaml.safe_load(config_yaml)
            
            # Update config manager (this is simplified - full implementation would update all fields)
            if 'game' in config_dict and 'detection_interval' in config_dict['game']:
                self.interval_var.set(config_dict['game']['detection_interval'])
            
            # Save to file
            config_manager.save_config()
            
            self._update_status("💾 Configuration saved")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save config: {e}")
    
    def _open_config_file(self):
        """Open the configuration file in default editor"""
        try:
            config_path = config_manager.config_file
            if config_path.exists():
                # Open with default application
                if hasattr(subprocess, 'run'):
                    subprocess.run(['notepad.exe', str(config_path)], check=False)
                else:
                    webbrowser.open(str(config_path))
            else:
                messagebox.showwarning("File Not Found", f"Config file not found: {config_path}")
                
        except Exception as e:
            messagebox.showerror("Open Error", f"Failed to open config file: {e}")
    
    def _save_screenshot(self):
        """Save the current screenshot"""
        if self.current_screenshot is None:
            messagebox.showwarning("No Screenshot", "No screenshot available to save.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            
            if file_path:
                cv2.imwrite(file_path, self.current_screenshot)
                self._update_status(f"📸 Screenshot saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save screenshot: {e}")
    
    def _test_connection(self):
        """Test the current connection"""
        if not self.device_manager or not self.device_manager.adb.connected:
            messagebox.showwarning("No Connection", "Please connect to a device first.")
            return
        
        def test_thread():
            try:
                self._update_status("🧪 Testing connection...")
                
                # Test screenshot
                screenshot = self.device_manager.adb.take_screenshot()
                if screenshot is not None:
                    # Test app detection
                    foreground_app = self.device_manager.adb.get_foreground_app()
                    
                    # Test game detection
                    if self.game_detector:
                        is_running, package = self.game_detector.is_lineage2m_running()
                        
                        test_results = {
                            'screenshot': True,
                            'app_detection': foreground_app is not None,
                            'game_detection': is_running,
                            'foreground_app': foreground_app,
                            'game_package': package if is_running else None
                        }
                    else:
                        test_results = {
                            'screenshot': True,
                            'app_detection': foreground_app is not None,
                            'game_detection': None,
                            'foreground_app': foreground_app,
                            'game_package': None
                        }
                    
                    self.message_queue.put({
                        'type': 'test_results',
                        'results': test_results
                    })
                else:
                    raise BotError("Screenshot test failed")
                    
            except Exception as e:
                logger.error(f"Connection test error: {e}")
                self.message_queue.put({
                    'type': 'error',
                    'message': f"Connection test failed: {e}"
                })
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """Lineage 2M Bot - Advanced Automation Framework
        
Version: 1.0.0
Author: Lineage2M Bot Team

A comprehensive automation framework for Lineage 2M mobile game using ADB, computer vision, and advanced bot capabilities.

Features:
• Device discovery and management
• Game state detection and monitoring
• Screenshot capture and analysis
• Modern GUI interface
• Configurable automation settings
• Advanced logging system

For support and documentation, visit the project repository."""
        
        messagebox.showinfo("About Lineage 2M Bot", about_text)
    
    def _process_message(self, message):
        """Process messages from background threads"""
        msg_type = message.get('type')
        
        if msg_type == 'devices_discovered':
            self._update_device_list(message['devices'])
            self._update_status(f"✅ Found {len(message['devices'])} device(s)")
            
        elif msg_type == 'device_connected':
            device = message['device']
            self.connection_status.configure(text=f"🟢 Connected: {device['id']}")
            self._update_status(f"✅ Connected to {device['id']}")
            
            # Enable bot controls
            self.start_bot_btn.configure(state="normal")
            self.screenshot_btn.configure(state="normal")
            
        elif msg_type == 'bot_started':
            self.start_bot_btn.configure(state="disabled")
            self.stop_bot_btn.configure(state="normal")
            
        elif msg_type == 'bot_stopped':
            self.start_bot_btn.configure(state="normal")
            self.stop_bot_btn.configure(state="disabled")
            self._update_status("⏹️ Bot stopped")
            
        elif msg_type == 'game_status_update':
            self._update_game_status(message)
            
        elif msg_type == 'screenshot_taken':
            self._update_status("📸 Screenshot captured")
            
        elif msg_type == 'test_results':
            self._show_test_results(message['results'])
            
        elif msg_type == 'error':
            messagebox.showerror("Error", message['message'])
            self._update_status(f"❌ Error: {message['message']}")
    
    def _update_device_list(self, devices):
        """Update the device list display"""
        # Clear existing items
        for item in self.device_tree.get_children():
            self.device_tree.delete(item)
        
        # Add devices
        for device in devices:
            status_icon = "🟢" if device['status'] == 'connected' else "🟡"
            
            # Game status information
            game_status = device.get('game_status', {})
            game_info = "Not installed"
            
            if game_status.get('running'):
                running_count = len(game_status.get('running_packages', []))
                game_info = f"🎮 Running ({running_count})"
            elif game_status.get('installed'):
                installed_count = len(game_status.get('installed_packages', []))
                game_info = f"📱 Installed ({installed_count})"
            
            self.device_tree.insert(
                "",
                "end",
                text=device['id'],
                values=(
                    device.get('type', 'Unknown'),
                    device.get('model', 'Unknown'),
                    device.get('android_version', 'Unknown'),
                    device.get('resolution', 'Unknown'),
                    f"{status_icon} {device['status']}",
                    game_info
                )
            )
    
    def _update_game_status(self, message):
        """Update the game status display"""
        if message['running']:
            status_text = f"🎮 Game Running: {message['package']}\n"
            if message['state']:
                state = message['state']
                status_text += f"Screen Size: {state.get('screen_size', 'Unknown')}\n"
                status_text += f"Menu State: {state.get('menu_state', 'Unknown')}\n"
                status_text += f"UI Elements: {state.get('ui_elements', 0)}\n"
                status_text += f"Last Update: {state.get('timestamp', 'Unknown')}\n"
        else:
            status_text = "❌ Lineage 2M not detected\n"
            
            # Show current foreground app
            foreground_app = self.device_manager.adb.get_foreground_app()
            if foreground_app:
                status_text += f"Current App: {foreground_app}\n"
        
        # Update text area
        self.game_status_text.delete("1.0", tk.END)
        self.game_status_text.insert("1.0", status_text)
        
        # Add to log
        self._add_log(status_text.strip())
    
    def _update_screenshot_display(self, screenshot):
        """Update the screenshot display"""
        try:
            # Resize screenshot to fit canvas
            canvas_width = self.screenshot_canvas.winfo_width()
            canvas_height = self.screenshot_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Convert OpenCV to PIL
                screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(screenshot_rgb)
                
                # Calculate aspect ratio
                img_width, img_height = pil_image.size
                canvas_ratio = canvas_width / canvas_height
                img_ratio = img_width / img_height
                
                if img_ratio > canvas_ratio:
                    # Image is wider
                    new_width = canvas_width - 20
                    new_height = int(new_width / img_ratio)
                else:
                    # Image is taller
                    new_height = canvas_height - 20
                    new_width = int(new_height * img_ratio)
                
                # Resize image
                resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(resized_image)
                
                # Update canvas
                self.screenshot_canvas.delete("all")
                self.screenshot_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=photo,
                    anchor="center"
                )
                
                # Keep a reference to prevent garbage collection
                self.screenshot_canvas.image = photo
                
        except Exception as e:
            logger.error(f"Error updating screenshot display: {e}")
    
    def _show_test_results(self, results):
        """Show connection test results"""
        result_text = "🧪 Connection Test Results:\n\n"
        result_text += f"Screenshot: {'✅' if results['screenshot'] else '❌'}\n"
        result_text += f"App Detection: {'✅' if results['app_detection'] else '❌'}\n"
        
        if results['game_detection'] is not None:
            result_text += f"Game Detection: {'✅' if results['game_detection'] else '❌'}\n"
        
        if results['foreground_app']:
            result_text += f"Current App: {results['foreground_app']}\n"
        
        if results['game_package']:
            result_text += f"Game Package: {results['game_package']}\n"
        
        messagebox.showinfo("Test Results", result_text)
        self._update_status("🧪 Connection test completed")
    
    def _add_log(self, text):
        """Add text to the log display"""
        try:
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.log_text.see(tk.END)  # Scroll to bottom
        except Exception as e:
            logger.error(f"Error adding log: {e}")
    
    def _update_status(self, text):
        """Update the status bar"""
        try:
            self.status_label.configure(text=text)
            self._add_log(text)
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def _show_progress(self):
        """Show progress bar"""
        try:
            self.progress_bar.pack(side="right", padx=10, pady=5)
            self.progress_bar.start()
        except Exception as e:
            logger.error(f"Error showing progress: {e}")
    
    def _hide_progress(self):
        """Hide progress bar"""
        try:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
        except Exception as e:
            logger.error(f"Error hiding progress: {e}")

# Import time module for the handlers
import time