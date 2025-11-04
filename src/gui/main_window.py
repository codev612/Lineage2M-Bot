"""
Main GUI Application - Modern desktop interface for Lineage 2M Bot
Provides comprehensive bot management through an intuitive graphical interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json

from ..core.device_manager import DeviceManager
from ..core.multi_device_manager import MultiDeviceManager
from ..modules.game_detector import GameDetector
from ..utils.config import config_manager
from ..utils.logger import get_logger
from ..utils.exceptions import BotError
from ..utils.device_state_monitor import device_state_monitor

logger = get_logger(__name__)

# Set customtkinter theme and color
ctk.set_appearance_mode("dark")  # "light" or "dark"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Import GUI handlers after setting up the theme
from .gui_handlers import GUIEventHandlers

class MainWindow(GUIEventHandlers):
    """
    Main GUI window for Lineage 2M Bot
    Provides comprehensive bot management interface
    """
    
    def __init__(self):
        """Initialize the main GUI window"""
        self.root = ctk.CTk()
        self.root.title("Lineage 2M Bot - Advanced Automation Framework")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Initialize components
        self.config = config_manager.get_config()
        self.device_manager = DeviceManager()
        self.multi_device_manager = MultiDeviceManager()
        self.game_detector = None
        self.bot_running = False
        
        # GUI state - Multi-device support
        self.selected_device = None  # Keep for compatibility
        self.multi_device_mode = True  # Enable multi-device mode by default
        self.devices_list = []
        self.selected_device = None
        
        # Threading
        self.message_queue = queue.Queue()
        self.screenshot_queue = queue.Queue()
        
        # Create GUI
        self._setup_gui()
        self._setup_menu()
        self._start_update_thread()
        
        # Restore saved devices on startup
        self._restore_saved_devices()
        
        # Start device state monitoring
        device_state_monitor.set_update_callback(self._update_device_states)
        device_state_monitor.start_monitoring(interval=2.0)
        
        logger.info("GUI initialized successfully")
    
    def _setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header
        self._create_header()
        
        # Create main content area with tabs
        self._create_tabview()
        
        # Create status bar
        self._create_status_bar()
    
    def _create_header(self):
        """Create the header section"""
        header_frame = ctk.CTkFrame(self.main_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="🎮 Lineage 2M Bot", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        # Connection status
        self.connection_status = ctk.CTkLabel(
            header_frame,
            text="🔴 Disconnected",
            font=ctk.CTkFont(size=14)
        )
        self.connection_status.pack(side="right", padx=20, pady=15)
    
    def _create_tabview(self):
        """Create the main tabbed interface"""
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create tabs
        self.device_tab = self.tabview.add("🔌 Device Manager")
        self.bot_tab = self.tabview.add("🤖 Bot Control")
        self.monitor_tab = self.tabview.add("📊 Monitor")
        self.settings_tab = self.tabview.add("⚙️ Settings")
        
        # Setup each tab
        self._setup_device_tab()
        self._setup_bot_tab()
        self._setup_monitor_tab()
        self._setup_settings_tab()
    
    def _setup_device_tab(self):
        """Setup the device management tab"""
        # Device list with integrated controls
        list_frame = ctk.CTkFrame(self.device_tab)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header with title and buttons
        header_frame = ctk.CTkFrame(list_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        list_label = ctk.CTkLabel(
            header_frame,
            text="📱 Available Devices",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        list_label.pack(side="left", pady=10)
        
        # Control buttons on the right side of header
        button_frame = ctk.CTkFrame(header_frame)
        button_frame.pack(side="right", padx=10, pady=5)
        
        self.add_device_btn = ctk.CTkButton(
            button_frame,
            text="➕ Add Device",
            command=self._add_device_manually,
            width=110
        )
        self.add_device_btn.pack(side="left", padx=2, pady=5)
        
        # Multi-device control buttons
        self.select_all_btn = ctk.CTkButton(
            button_frame,
            text="☑️ Select All",
            command=self._select_all_devices,
            width=100
        )
        self.select_all_btn.pack(side="left", padx=2, pady=5)
        
        self.disconnect_all_btn = ctk.CTkButton(
            button_frame,
            text="🚫 Disconnect All",
            command=self._disconnect_all_devices,
            width=120,
            state="disabled"
        )
        self.disconnect_all_btn.pack(side="left", padx=2, pady=5)
        
        # Create treeview for device list with checkboxes
        self.device_tree = ttk.Treeview(
            list_frame,
            columns=("Select", "Type", "Model", "Android", "Resolution", "Status", "Game", "Connection"),
            show="tree headings",
            height=8
        )
        
        # Configure columns
        self.device_tree.heading("#0", text="Device ID")
        self.device_tree.heading("Select", text="☐")
        self.device_tree.heading("Type", text="Type")
        self.device_tree.heading("Model", text="Model")
        self.device_tree.heading("Android", text="Android")
        self.device_tree.heading("Resolution", text="Resolution")
        self.device_tree.heading("Status", text="Status")
        self.device_tree.heading("Game", text="Lineage 2M")
        self.device_tree.heading("Connection", text="Connection")
        
        # Column widths
        self.device_tree.column("#0", width=140)
        self.device_tree.column("Select", width=50)
        self.device_tree.column("Type", width=100)
        self.device_tree.column("Model", width=90)
        self.device_tree.column("Android", width=70)
        self.device_tree.column("Resolution", width=100)
        self.device_tree.column("Status", width=100)
        self.device_tree.column("Game", width=120)
        self.device_tree.column("Connection", width=100)
        
        # Bind selection event
        self.device_tree.bind("<<TreeviewSelect>>", self._on_device_select)
        
        # Add scrollbar
        tree_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.device_tree.yview)
        self.device_tree.configure(yscroll=tree_scrollbar.set)
        
        # Pack treeview and scrollbar
        tree_frame = tk.Frame(list_frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.device_tree.pack(side="left", fill="both", expand=True)
        tree_scrollbar.pack(side="right", fill="y")
    
    def _setup_bot_tab(self):
        """Setup the bot control tab"""
        # Multi-device control section
        device_control_frame = ctk.CTkFrame(self.bot_tab)
        device_control_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.device_control_label = ctk.CTkLabel(
            device_control_frame,
            text="🎮 Per-Device Control",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.device_control_label.pack(pady=10)
        
        # Scrollable frame for device controls
        self.device_control_scroll = ctk.CTkScrollableFrame(
            device_control_frame,
            height=400
        )
        self.device_control_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Global controls section
        global_control_frame = ctk.CTkFrame(self.bot_tab)
        global_control_frame.pack(fill="x", padx=10, pady=10)
        
        global_control_label = ctk.CTkLabel(
            global_control_frame,
            text="🤖 Global Controls (Selected Devices)",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        global_control_label.pack(pady=10)
        
        # Global control buttons
        global_btn_frame = ctk.CTkFrame(global_control_frame)
        global_btn_frame.pack(fill="x", padx=10, pady=5)
        
        self.start_all_btn = ctk.CTkButton(
            global_btn_frame,
            text="▶️ Start Selected",
            command=self._start_all_bots,
            width=140,
            state="disabled"
        )
        self.start_all_btn.pack(side="left", padx=5, pady=10)
        
        self.stop_all_btn = ctk.CTkButton(
            global_btn_frame,
            text="⏹️ Stop Selected",
            command=self._stop_all_bots,
            width=140,
            state="disabled"
        )
        self.stop_all_btn.pack(side="left", padx=5, pady=10)
        
        self.screenshot_all_btn = ctk.CTkButton(
            global_btn_frame,
            text="📸 Screenshot Selected",
            command=self._take_screenshot_all,
            width=150,
            state="disabled"
        )
        self.screenshot_all_btn.pack(side="left", padx=5, pady=10)
        
        # Device control widgets storage
        self.device_control_widgets = {}
        
        # Bot settings
        settings_frame = ctk.CTkFrame(self.bot_tab)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        settings_label = ctk.CTkLabel(
            settings_frame,
            text="⚙️ Bot Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        settings_label.pack(pady=(10, 5))
        
        # Detection interval
        interval_frame = ctk.CTkFrame(settings_frame)
        interval_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(interval_frame, text="Detection Interval (seconds):").pack(side="left", padx=10, pady=10)
        
        self.interval_var = tk.DoubleVar(value=self.config.game.detection_interval)
        self.interval_scale = ctk.CTkSlider(
            interval_frame,
            from_=1.0,
            to=10.0,
            variable=self.interval_var,
            width=200
        )
        self.interval_scale.pack(side="left", padx=10, pady=10)
        
        self.interval_label = ctk.CTkLabel(interval_frame, text=f"{self.interval_var.get():.1f}s")
        self.interval_label.pack(side="left", padx=10, pady=10)
        
        # Update label when slider changes
        self.interval_scale.configure(command=self._update_interval_label)
    
    def _create_device_control_widget(self, device):
        """Create control widget for a specific device"""
        device_id = device['id']
        
        # Prevent duplicate widget creation
        if device_id in self.device_control_widgets:
            logger.warning(f"Widget already exists for device {device_id}, skipping creation")
            return
        
        # Create frame for this device
        device_frame = ctk.CTkFrame(self.device_control_scroll)
        device_frame.pack(fill="x", padx=5, pady=5)
        
        # Device header
        header_frame = ctk.CTkFrame(device_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Device info - Show device address prominently
        device_address = device_id  # This is the device address (e.g., 127.0.0.1:5555)
        device_info = f"📱 Device: {device_address}"
        if device.get('model') != 'Unknown':
            device_info += f" | {device['model']}"
        
        game_status = device.get('game_status', {})
        if game_status.get('running'):
            device_info += " 🎮"
        elif game_status.get('installed'):
            device_info += " 📱"
        
        device_label = ctk.CTkLabel(
            header_frame,
            text=device_info,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        device_label.pack(side="left", padx=10, pady=5)
        
        # Also show device address in a smaller label for clarity
        address_label = ctk.CTkLabel(
            header_frame,
            text=f"Address: {device_address}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        address_label.pack(side="left", padx=5, pady=5)
        
        # Connection status
        connected_devices = self.multi_device_manager.get_connected_devices()
        is_connected = device_id in connected_devices
        status_text = "🟢 Connected" if is_connected else "🔴 Disconnected"
        
        status_label = ctk.CTkLabel(
            header_frame,
            text=status_text,
            font=ctk.CTkFont(size=12)
        )
        status_label.pack(side="right", padx=10, pady=5)
        
        # Control buttons
        control_frame = ctk.CTkFrame(device_frame)
        control_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        # Bot control buttons
        start_btn = ctk.CTkButton(
            control_frame,
            text="▶️ Start Bot",
            command=lambda: self._start_device_bot(device_id),
            width=100,
            state="normal" if is_connected else "disabled"
        )
        start_btn.pack(side="left", padx=5, pady=5)
        
        stop_btn = ctk.CTkButton(
            control_frame,
            text="⏹️ Stop Bot",
            command=lambda: self._stop_device_bot(device_id),
            width=100,
            state="disabled"
        )
        stop_btn.pack(side="left", padx=5, pady=5)
        
        screenshot_btn = ctk.CTkButton(
            control_frame,
            text="📸 Screenshot",
            command=lambda: self._take_device_screenshot(device_id),
            width=100,
            state="normal" if is_connected else "disabled"
        )
        screenshot_btn.pack(side="left", padx=5, pady=5)
        
        # Test action button - automated swipe and tap sequence
        # Use default argument to capture device_id properly in closure
        test_btn = ctk.CTkButton(
            control_frame,
            text="🧪 Test",
            command=lambda dev_id=device_id: self._device_test_actions(dev_id),
            width=80,
            state="normal" if is_connected else "disabled"
        )
        test_btn.pack(side="left", padx=5, pady=5)
        
        # Check game status button
        check_game_btn = ctk.CTkButton(
            control_frame,
            text="🔍 Check Game",
            command=lambda dev_id=device_id: self._check_game_status(dev_id),
            width=100,
            state="normal" if is_connected else "disabled"
        )
        check_game_btn.pack(side="left", padx=5, pady=5)
        
        # State monitoring labels
        state_frame = ctk.CTkFrame(device_frame)
        state_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        # Bot state label
        bot_state_label = ctk.CTkLabel(
            state_frame,
            text="🤖 Bot: Stopped",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        bot_state_label.pack(fill="x", padx=5, pady=2)
        
        # Game state label
        game_state_label = ctk.CTkLabel(
            state_frame,
            text="🎮 Game: Not Running",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        game_state_label.pack(fill="x", padx=5, pady=2)
        
        # Store widget references for later updates
        self.device_control_widgets[device_id] = {
            'frame': device_frame,
            'status_label': status_label,
            'bot_state_label': bot_state_label,
            'game_state_label': game_state_label,
            'start_btn': start_btn,
            'stop_btn': stop_btn,
            'screenshot_btn': screenshot_btn,
            'test_btn': test_btn,
            'check_game_btn': check_game_btn,
            'running': False
        }
    
    def _refresh_device_control_widgets(self):
        """Refresh device control widgets for selected devices only"""
        # Schedule the refresh to avoid widget destruction race conditions
        self.root.after_idle(self._do_refresh_device_control_widgets)
    
    def _do_refresh_device_control_widgets(self):
        """Actually perform the widget refresh (called via after_idle)"""
        try:
            # Clear existing widgets safely
            widgets_to_destroy = list(self.device_control_widgets.values())
            logger.info(f"Refreshing device control widgets: destroying {len(widgets_to_destroy)} existing widgets")
            self.device_control_widgets.clear()
            
            # Destroy widgets on next idle cycle
            for widgets in widgets_to_destroy:
                if widgets['frame'].winfo_exists():
                    try:
                        widgets['frame'].destroy()
                    except Exception as e:
                        logger.warning(f"Error destroying widget frame: {e}")
            
            # Get selected devices from the device tree
            selected_devices = self._get_selected_devices_from_tree()
            connected_devices = self.multi_device_manager.get_connected_devices()
            
            logger.info(f"Found {len(selected_devices)} selected device(s), {len(connected_devices)} connected device(s)")
            
            # Deduplicate devices by device_id to prevent duplicate widgets
            seen_device_ids = set()
            unique_selected_devices = []
            for device in selected_devices:
                device_id = device['id']
                if device_id not in seen_device_ids:
                    seen_device_ids.add(device_id)
                    unique_selected_devices.append(device)
                else:
                    logger.warning(f"Duplicate device in selected devices: {device_id}, skipping")
            
            logger.info(f"After deduplication: {len(unique_selected_devices)} unique device(s)")
            
            # Create widgets only for selected devices that are connected
            selected_connected_devices = []
            for device in unique_selected_devices:
                device_id = device['id']
                if device_id in connected_devices:
                    selected_connected_devices.append(device)
                    logger.info(f"Creating widget for device: {device_id}")
                    self._create_device_control_widget(device)
                else:
                    logger.debug(f"Device {device_id} is selected but not connected, skipping widget creation")
            
            logger.info(f"Created widgets for {len(selected_connected_devices)} device(s)")
            
            # Update global button states based on selected connected devices
            has_selected_connected = len(selected_connected_devices) > 0
            self.start_all_btn.configure(state="normal" if has_selected_connected else "disabled")
            self.stop_all_btn.configure(state="normal" if has_selected_connected else "disabled")
            self.screenshot_all_btn.configure(state="normal" if has_selected_connected else "disabled")
            
            # Update the label to show selected vs connected count
            if selected_connected_devices:
                self.device_control_label.configure(
                    text=f"🎮 Per-Device Control ({len(selected_connected_devices)} selected)"
                )
            else:
                self.device_control_label.configure(text="🎮 Per-Device Control (No devices selected)")
        
        except Exception as e:
            logger.error(f"Error refreshing device control widgets: {e}")
            # Fallback: clear the label
            if hasattr(self, 'device_control_label'):
                self.device_control_label.configure(text="🎮 Per-Device Control (Error)")
    
    # _get_selected_devices_from_tree method is inherited from GUIEventHandlers
    
    def _setup_monitor_tab(self):
        """Setup the monitoring tab"""
        # Screenshot viewer
        screenshot_frame = ctk.CTkFrame(self.monitor_tab)
        screenshot_frame.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        
        screenshot_label = ctk.CTkLabel(
            screenshot_frame,
            text="📸 Live Screenshot",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        screenshot_label.pack(pady=10)
        
        # Screenshot display
        self.screenshot_canvas = tk.Canvas(
            screenshot_frame,
            bg="black",
            width=300,
            height=400
        )
        self.screenshot_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Log viewer
        log_frame = ctk.CTkFrame(self.monitor_tab)
        log_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="📝 Bot Logs",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        log_label.pack(pady=10)
        
        # Log text area
        self.log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Log control buttons
        log_btn_frame = ctk.CTkFrame(log_frame)
        log_btn_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            log_btn_frame,
            text="🗑️ Clear Logs",
            command=self._clear_logs,
            width=100
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            log_btn_frame,
            text="💾 Save Logs",
            command=self._save_logs,
            width=100
        ).pack(side="left", padx=5, pady=5)
    
    def _setup_settings_tab(self):
        """Setup the settings tab"""
        # Configuration section
        config_frame = ctk.CTkFrame(self.settings_tab)
        config_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        config_label = ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        config_label.pack(pady=10)
        
        # Configuration display/editor
        self.config_text = ctk.CTkTextbox(
            config_frame,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.config_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load current configuration
        self._load_config_display()
        
        # Config buttons
        config_btn_frame = ctk.CTkFrame(config_frame)
        config_btn_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            config_btn_frame,
            text="🔄 Reload Config",
            command=self._reload_config,
            width=120
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            config_btn_frame,
            text="💾 Save Config",
            command=self._save_config,
            width=120
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            config_btn_frame,
            text="📁 Open Config File",
            command=self._open_config_file,
            width=140
        ).pack(side="left", padx=5, pady=5)
    
    def _setup_menu(self):
        """Setup the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Config...", command=self._open_config_file)
        file_menu.add_command(label="Save Screenshot...", command=self._save_screenshot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Device Discovery", command=self._discover_devices)
        tools_menu.add_command(label="Test Connection", command=self._test_connection)
        tools_menu.add_command(label="Take Screenshot", command=self._take_screenshot)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready - Select a device to begin",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        # Progress bar (hidden by default)
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.set(0)
        # Don't pack initially - will show when needed
    
    def _start_update_thread(self):
        """Start the GUI update thread"""
        def update_loop():
            while True:
                try:
                    # Process message queue
                    while not self.message_queue.empty():
                        message = self.message_queue.get_nowait()
                        self._process_message(message)
                    
                    # Process screenshot queue
                    while not self.screenshot_queue.empty():
                        screenshot = self.screenshot_queue.get_nowait()
                        self._update_screenshot_display(screenshot)
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in GUI update thread: {e}")
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    # Event handlers and GUI methods will continue in the next part...
    
    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
    
    def _update_device_states(self):
        """Update device state displays (called by state monitor)"""
        try:
            # Schedule GUI update on main thread
            self.root.after(0, self._do_update_device_states)
        except Exception as e:
            logger.error(f"Error scheduling device state update: {e}")
    
    def _do_update_device_states(self):
        """Actually update device state displays (runs on main thread)"""
        try:
            # Get all device states
            all_states = device_state_monitor.get_all_states()
            
            # Update each device widget
            for device_id in all_states.keys():
                if device_id in self.device_control_widgets:
                    widgets = self.device_control_widgets[device_id]
                    
                    # Get state summary
                    summary = device_state_monitor.get_state_summary(device_id)
                    
                    # Update bot state label (summary already includes emoji)
                    if 'bot_state_label' in widgets and widgets['bot_state_label'].winfo_exists():
                        widgets['bot_state_label'].configure(text=f"🤖 Bot: {summary['bot_status']}")
                    
                    # Update game state label (summary already includes emoji)
                    if 'game_state_label' in widgets and widgets['game_state_label'].winfo_exists():
                        # Format game status more clearly
                        game_status_text = summary['game_status']
                        # Remove duplicate emoji if present
                        if game_status_text.startswith('🎮'):
                            game_status_text = game_status_text[1:].strip()
                        elif game_status_text.startswith('⏸️'):
                            game_status_text = game_status_text[1:].strip()
                        
                        widgets['game_state_label'].configure(text=f"🎮 Game: {game_status_text}")
        except Exception as e:
            logger.error(f"Error updating device states: {e}", exc_info=True)
    
    def _on_closing(self):
        """Handle window closing"""
        if self.bot_running:
            if messagebox.askokcancel("Quit", "Bot is running. Do you want to stop it and quit?"):
                self._stop_bot()
        
        # Stop device state monitoring
        device_state_monitor.stop_monitoring()
        
        self.root.destroy()

# Additional GUI methods will be added in the next file...