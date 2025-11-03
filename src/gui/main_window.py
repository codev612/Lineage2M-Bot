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
from ..modules.game_detector import GameDetector
from ..utils.config import config_manager
from ..utils.logger import get_logger
from ..utils.exceptions import BotError

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
        self.game_detector = None
        self.bot_running = False
        
        # GUI state
        self.current_screenshot = None
        self.devices_list = []
        self.selected_device = None
        
        # Threading
        self.message_queue = queue.Queue()
        self.screenshot_queue = queue.Queue()
        
        # Create GUI
        self._setup_gui()
        self._setup_menu()
        self._start_update_thread()
        
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
        # Device discovery section
        discovery_frame = ctk.CTkFrame(self.device_tab)
        discovery_frame.pack(fill="x", padx=10, pady=10)
        
        discovery_label = ctk.CTkLabel(
            discovery_frame,
            text="📱 Device Discovery",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        discovery_label.pack(pady=10)
        
        # Discovery buttons
        button_frame = ctk.CTkFrame(discovery_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        self.discover_btn = ctk.CTkButton(
            button_frame,
            text="🔍 Discover Devices",
            command=self._discover_devices,
            width=150
        )
        self.discover_btn.pack(side="left", padx=5, pady=10)
        
        self.refresh_btn = ctk.CTkButton(
            button_frame,
            text="🔄 Refresh",
            command=self._refresh_devices,
            width=100
        )
        self.refresh_btn.pack(side="left", padx=5, pady=10)
        
        self.connect_btn = ctk.CTkButton(
            button_frame,
            text="🔗 Connect",
            command=self._connect_to_device,
            width=100,
            state="disabled"
        )
        self.connect_btn.pack(side="left", padx=5, pady=10)
        
        # Device list
        list_frame = ctk.CTkFrame(self.device_tab)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        list_label = ctk.CTkLabel(
            list_frame,
            text="📋 Available Devices",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        list_label.pack(pady=(10, 5))
        
        # Create treeview for device list
        self.device_tree = ttk.Treeview(
            list_frame,
            columns=("Type", "Model", "Android", "Resolution", "Status", "Game"),
            show="tree headings",
            height=8
        )
        
        # Configure columns
        self.device_tree.heading("#0", text="Device ID")
        self.device_tree.heading("Type", text="Type")
        self.device_tree.heading("Model", text="Model")
        self.device_tree.heading("Android", text="Android")
        self.device_tree.heading("Resolution", text="Resolution")
        self.device_tree.heading("Status", text="Status")
        self.device_tree.heading("Game", text="Lineage 2M")
        
        # Column widths
        self.device_tree.column("#0", width=150)
        self.device_tree.column("Type", width=120)
        self.device_tree.column("Model", width=100)
        self.device_tree.column("Android", width=80)
        self.device_tree.column("Resolution", width=100)
        self.device_tree.column("Status", width=100)
        self.device_tree.column("Game", width=130)
        
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
        # Game status section
        status_frame = ctk.CTkFrame(self.bot_tab)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        status_label = ctk.CTkLabel(
            status_frame,
            text="🎮 Game Status",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        status_label.pack(pady=10)
        
        # Game info display
        self.game_status_text = ctk.CTkTextbox(
            status_frame,
            height=100,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.game_status_text.pack(fill="x", padx=10, pady=5)
        
        # Bot control section
        control_frame = ctk.CTkFrame(self.bot_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        control_label = ctk.CTkLabel(
            control_frame,
            text="🤖 Bot Controls",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        control_label.pack(pady=10)
        
        # Control buttons
        btn_frame = ctk.CTkFrame(control_frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        self.start_bot_btn = ctk.CTkButton(
            btn_frame,
            text="▶️ Start Bot",
            command=self._start_bot,
            width=120,
            state="disabled"
        )
        self.start_bot_btn.pack(side="left", padx=5, pady=10)
        
        self.stop_bot_btn = ctk.CTkButton(
            btn_frame,
            text="⏹️ Stop Bot",
            command=self._stop_bot,
            width=120,
            state="disabled"
        )
        self.stop_bot_btn.pack(side="left", padx=5, pady=10)
        
        self.screenshot_btn = ctk.CTkButton(
            btn_frame,
            text="📸 Screenshot",
            command=self._take_screenshot,
            width=120,
            state="disabled"
        )
        self.screenshot_btn.pack(side="left", padx=5, pady=10)
        
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
    
    def _on_closing(self):
        """Handle window closing"""
        if self.bot_running:
            if messagebox.askokcancel("Quit", "Bot is running. Do you want to stop it and quit?"):
                self._stop_bot()
                self.root.destroy()
        else:
            self.root.destroy()

# Additional GUI methods will be added in the next file...