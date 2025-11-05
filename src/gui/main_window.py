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
        # Limit screenshot queue to prevent memory buildup (max 3 screenshots)
        self.screenshot_queue = queue.Queue(maxsize=3)
        
        # Create GUI
        self._setup_gui()
        self._setup_menu()
        self._start_update_thread()
        
        # Restore saved devices on startup
        self._restore_saved_devices()
        
        # Start device state monitoring
        device_state_monitor.set_update_callback(self._update_device_states)
        device_state_monitor.start_monitoring(interval=2.0)
        
        # Update region tab device list after devices are loaded (with delay to ensure GUI is ready)
        self.root.after(2000, self._update_region_device_list)
        
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
        self.region_tab = self.tabview.add("📐 Region Config")
        self.settings_tab = self.tabview.add("⚙️ Settings")
        
        # Setup each tab
        self._setup_device_tab()
        self._setup_bot_tab()
        self._setup_monitor_tab()
        self._setup_region_tab()
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
    
    def _setup_region_tab(self):
        """Setup the region configuration tab with canvas for drawing regions"""
        # Main container
        main_frame = ctk.CTkFrame(self.region_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="📐 Region Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Left panel: Controls and region selection
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.pack(side="left", fill="y", padx=(0, 5), pady=5)
        left_panel.configure(width=250)
        
        # Device selection
        device_frame = ctk.CTkFrame(left_panel)
        device_frame.pack(fill="x", padx=10, pady=10)
        
        device_label = ctk.CTkLabel(device_frame, text="Device:", font=ctk.CTkFont(size=12, weight="bold"))
        device_label.pack(pady=5)
        
        self.region_device_var = tk.StringVar(value="Select device...")
        self.region_device_dropdown = ctk.CTkComboBox(
            device_frame,
            variable=self.region_device_var,
            values=["Select device..."],
            width=200,
            state="readonly"
        )
        self.region_device_dropdown.pack(pady=5)
        
        # Screenshot button
        screenshot_btn = ctk.CTkButton(
            device_frame,
            text="📸 Take Screenshot",
            command=self._take_screenshot_for_region,
            width=200
        )
        screenshot_btn.pack(pady=5)
        
        # Refresh device list button
        refresh_devices_btn = ctk.CTkButton(
            device_frame,
            text="🔄 Refresh Device List",
            command=self._update_region_device_list,
            width=200
        )
        refresh_devices_btn.pack(pady=5)
        
        # Region type selection
        region_type_frame = ctk.CTkFrame(left_panel)
        region_type_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        region_label = ctk.CTkLabel(region_type_frame, text="Region Type:", font=ctk.CTkFont(size=12, weight="bold"))
        region_label.pack(pady=5)
        
        # Scrollable frame for region types
        self.region_type_scroll = ctk.CTkScrollableFrame(
            region_type_frame,
            height=200
        )
        self.region_type_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.region_type_var = tk.StringVar(value="player")
        region_types = [
            ("player", "👤 Player Region"),
            ("quests", "📋 Quests Item Region"),
            ("control_buttons", "🎮 Control Buttons Region"),
            ("items", "💎 Items Region"),
            ("health_bar", "❤️ Health Bar Region"),
            ("mp_bar", "💙 MP Bar Region"),
            ("other", "🔧 Other Region")
        ]
        
        # Store region types for later use
        self.region_types_list = region_types.copy()
        
        # Store radio button references and tick labels for updating
        self.region_type_widgets = {}
        
        for region_type, label in region_types:
            # Create a frame to hold radio button and tick mark
            item_frame = ctk.CTkFrame(self.region_type_scroll)
            item_frame.pack(fill="x", padx=5, pady=2)
            
            rb = ctk.CTkRadioButton(
                item_frame,
                text=label,
                variable=self.region_type_var,
                value=region_type,
                command=lambda rt=region_type: self._set_region_type(rt)
            )
            rb.pack(side="left", padx=5)
            
            # Create tick label (initially empty)
            tick_label = ctk.CTkLabel(
                item_frame,
                text="",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#00ff00"  # Green color
            )
            tick_label.pack(side="left", padx=5)
            
            # Store references
            self.region_type_widgets[region_type] = {
                'radio': rb,
                'tick': tick_label,
                'frame': item_frame
            }
        
        # Check for saved regions and update ticks
        self._update_region_type_ticks()
        
        # Load custom region types from JSON files at startup
        self._load_custom_region_types_from_files()
        
        # Add custom region button
        add_custom_btn = ctk.CTkButton(
            region_type_frame,
            text="➕ Add Custom Region",
            command=self._add_custom_region_dialog,
            width=200
        )
        add_custom_btn.pack(pady=5)
        
        # Right panel: Canvas with floating buttons
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0), pady=5)
        
        # Canvas container with relative positioning for floating buttons
        canvas_container = ctk.CTkFrame(right_panel)
        canvas_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Import RegionCanvas
        from .region_config import RegionCanvas
        
        # Create canvas
        self.region_canvas = RegionCanvas(canvas_container, width=800, height=600)
        self.region_canvas.pack(fill="both", expand=True)
        
        # Store canvas container for drag bounds
        self.canvas_container = canvas_container
        
        # Floating toolbar container (positioned on top of canvas) - draggable
        # Create with transparent background
        self.floating_toolbar = ctk.CTkFrame(
            canvas_container, 
            corner_radius=8,
            fg_color="transparent",  # Transparent background
            border_width=0  # No border
        )
        
        # Initialize toolbar position immediately - use a visible default position
        # Position will be refined after container is sized
        self.floating_toolbar.place(x=10, y=10)  # Start at top-left so it's definitely visible
        self.floating_toolbar.lift()  # Ensure it's on top of canvas
        
        # Refine position after container is ready
        def init_toolbar_position():
            try:
                self.canvas_container.update_idletasks()
                container_width = self.canvas_container.winfo_width()
                container_height = self.canvas_container.winfo_height()
                
                if container_width > 0:
                    # Get actual toolbar width
                    self.floating_toolbar.update_idletasks()
                    toolbar_width = self.floating_toolbar.winfo_reqwidth()
                    
                    # Calculate proper position (top-right with padding)
                    initial_x = max(10, container_width - toolbar_width - 10)
                    initial_y = 10
                    
                    # Ensure position is within bounds
                    if initial_x + toolbar_width > container_width:
                        initial_x = container_width - toolbar_width - 10
                    if initial_x < 0:
                        initial_x = 10
                    
                    self.floating_toolbar.place(x=initial_x, y=initial_y)
                    self.floating_toolbar.lift()  # Ensure it's on top
                    logger.debug(f"Toolbar positioned at ({initial_x}, {initial_y}), size: {toolbar_width}x{self.floating_toolbar.winfo_reqheight()}")
            except Exception as e:
                logger.error(f"Error initializing toolbar position: {e}", exc_info=True)
                # Fallback: use default position
                self.floating_toolbar.place(x=600, y=10)
                self.floating_toolbar.lift()
        
        # Schedule refinement after GUI is ready (longer delay to ensure container is sized)
        self.root.after(300, init_toolbar_position)
        
        # Drag state variables
        self.toolbar_drag_start_x = 0
        self.toolbar_drag_start_y = 0
        self.toolbar_dragging = False
        self.toolbar_start_x = 0
        self.toolbar_start_y = 0
        self.toolbar_pressed_widget = None
        
        # Floating toolbar buttons - icon only (create buttons BEFORE making draggable)
        self._create_toolbar_button(
            self.floating_toolbar,
            "❌",
            "",
            self._clear_current_region
        )
        
        self._create_toolbar_button(
            self.floating_toolbar,
            "🔄",
            "",
            self._clear_all_regions
        )
        
        # Separator
        separator = ctk.CTkFrame(self.floating_toolbar, width=1, height=30, fg_color=("#404040", "#404040"))
        separator.pack(side="left", padx=5, pady=5)
        
        self._create_toolbar_button(
            self.floating_toolbar,
            "💾",
            "",
            self._save_regions
        )
        
        self._create_toolbar_button(
            self.floating_toolbar,
            "📂",
            "",
            self._load_regions
        )
        
        # Separator
        separator2 = ctk.CTkFrame(self.floating_toolbar, width=1, height=30, fg_color=("#404040", "#404040"))
        separator2.pack(side="left", padx=5, pady=5)
        
        # Zoom buttons
        self._create_toolbar_button(
            self.floating_toolbar,
            "🔍+",
            "",
            self._zoom_in
        )
        
        self._create_toolbar_button(
            self.floating_toolbar,
            "🔍-",
            "",
            self._zoom_out
        )
        
        self._create_toolbar_button(
            self.floating_toolbar,
            "🔍",
            "",
            self._zoom_fit
        )
        
        # Update toolbar to ensure it's visible
        self.floating_toolbar.update_idletasks()
        self.floating_toolbar.lift()
        
        # Make toolbar draggable (after buttons are created)
        self._setup_toolbar_drag()
        
        # Set initial region type
        self._set_region_type("player")
        
        # Update ticks and load regions when device selection changes
        if hasattr(self, 'region_device_var'):
            def on_device_change(*args):
                self._load_custom_region_types_from_files()
                self._update_region_type_ticks()
                self._auto_load_regions()
            self.region_device_var.trace('w', on_device_change)
    
    def _zoom_in(self):
        """Zoom in the canvas"""
        if hasattr(self, 'region_canvas'):
            self.region_canvas.zoom_in()
    
    def _zoom_out(self):
        """Zoom out the canvas"""
        if hasattr(self, 'region_canvas'):
            self.region_canvas.zoom_out()
    
    def _zoom_fit(self):
        """Reset zoom to fit canvas"""
        if hasattr(self, 'region_canvas'):
            self.region_canvas.zoom_fit()
    
    def _setup_toolbar_drag(self):
        """Setup drag and drop functionality for the floating toolbar"""
        def on_toolbar_button_press(event):
            """Start dragging"""
            self.toolbar_dragging = False  # Start as False, only become True if mouse moves
            # Get mouse position relative to root window
            self.toolbar_drag_start_x = event.x_root
            self.toolbar_drag_start_y = event.y_root
            # Get current toolbar position relative to container
            # Ensure we're using absolute positioning before getting coordinates
            if self.floating_toolbar.winfo_exists():
                # Get current position - if using relative positioning, convert to absolute
                try:
                    # Try to get current place info
                    place_info = self.floating_toolbar.place_info()
                    if 'relx' in place_info and place_info['relx']:
                        # Convert relative to absolute
                        self.canvas_container.update_idletasks()
                        container_width = self.canvas_container.winfo_width()
                        container_height = self.canvas_container.winfo_height()
                        relx = float(place_info.get('relx', 0))
                        rely = float(place_info.get('rely', 0))
                        anchor = place_info.get('anchor', 'nw')
                        x_offset = int(place_info.get('x', 0))
                        y_offset = int(place_info.get('y', 0))
                        
                        # Calculate absolute position based on anchor
                        if 'e' in anchor:
                            self.toolbar_start_x = int(container_width * relx) + x_offset
                        else:
                            self.toolbar_start_x = int(container_width * relx) + x_offset
                        
                        if 's' in anchor:
                            self.toolbar_start_y = int(container_height * rely) + y_offset
                        else:
                            self.toolbar_start_y = int(container_height * rely) + y_offset
                        
                        # Switch to absolute positioning immediately
                        self.floating_toolbar.place(x=self.toolbar_start_x, y=self.toolbar_start_y)
                    else:
                        # Already using absolute positioning
                        self.toolbar_start_x = self.floating_toolbar.winfo_x()
                        self.toolbar_start_y = self.floating_toolbar.winfo_y()
                except:
                    # Fallback: get position directly
                    self.toolbar_start_x = self.floating_toolbar.winfo_x()
                    self.toolbar_start_y = self.floating_toolbar.winfo_y()
            else:
                self.toolbar_start_x = 0
                self.toolbar_start_y = 0
            
            # Store which widget was pressed
            self.toolbar_pressed_widget = event.widget
            # Change cursor to move cursor (only on toolbar, not canvas)
            self.floating_toolbar.configure(cursor="fleur")
        
        def on_toolbar_button_motion(event):
            """Handle dragging"""
            # Check if mouse moved enough to consider it a drag (threshold: 5 pixels)
            dx = abs(event.x_root - self.toolbar_drag_start_x)
            dy = abs(event.y_root - self.toolbar_drag_start_y)
            
            if dx > 5 or dy > 5:
                if not self.toolbar_dragging:
                    # Just started dragging - ensure cursor is set
                    self.toolbar_dragging = True
                    self.floating_toolbar.configure(cursor="fleur")
            
            if self.toolbar_dragging:
                # Calculate movement delta
                dx = event.x_root - self.toolbar_drag_start_x
                dy = event.y_root - self.toolbar_drag_start_y
                
                # Calculate new position relative to container
                new_x = self.toolbar_start_x + dx
                new_y = self.toolbar_start_y + dy
                
                # Get container bounds (cache if possible to reduce updates)
                container_width = self.canvas_container.winfo_width()
                container_height = self.canvas_container.winfo_height()
                
                # Get toolbar size (cache if possible)
                toolbar_width = self.floating_toolbar.winfo_reqwidth()
                toolbar_height = self.floating_toolbar.winfo_reqheight()
                
                # Constrain to container bounds
                new_x = max(0, min(new_x, container_width - toolbar_width))
                new_y = max(0, min(new_y, container_height - toolbar_height))
                
                # Update position directly without place_forget to avoid blinking
                # Ensure toolbar remains visible and on top
                if self.floating_toolbar.winfo_exists():
                    self.floating_toolbar.place(x=new_x, y=new_y)
                    self.floating_toolbar.lift()  # Ensure it's on top
        
        def on_toolbar_button_release(event):
            """Stop dragging"""
            # If we were dragging, don't trigger button click
            was_dragging = self.toolbar_dragging
            
            # Reset dragging state
            self.toolbar_dragging = False
            
            # Restore normal cursor (only on toolbar)
            self.floating_toolbar.configure(cursor="")
            
            # If we weren't dragging and released on a button, trigger its command
            if not was_dragging and self.toolbar_pressed_widget:
                # Check if the pressed widget is a button
                if isinstance(self.toolbar_pressed_widget, ctk.CTkButton):
                    # Get the button's command
                    button_cmd = self.toolbar_pressed_widget.cget("command")
                    if button_cmd:
                        # Execute command after a small delay to ensure state is reset
                        self.root.after(10, button_cmd)
            
            self.toolbar_pressed_widget = None
        
        # Bind drag events to toolbar frame
        self.floating_toolbar.bind("<Button-1>", on_toolbar_button_press)
        self.floating_toolbar.bind("<B1-Motion>", on_toolbar_button_motion)
        self.floating_toolbar.bind("<ButtonRelease-1>", on_toolbar_button_release)
        
        # Also bind to toolbar buttons to allow dragging from anywhere
        # But buttons will still work if clicked without dragging
        def create_button_wrapper(btn):
            """Create wrapper to handle button clicks and drags"""
            def button_press_handler(event):
                on_toolbar_button_press(event)
            
            def button_motion_handler(event):
                on_toolbar_button_motion(event)
            
            def button_release_handler(event):
                on_toolbar_button_release(event)
            
            btn.bind("<Button-1>", button_press_handler)
            btn.bind("<B1-Motion>", button_motion_handler)
            btn.bind("<ButtonRelease-1>", button_release_handler)
        
        # Bind drag handlers to all buttons (will be called after buttons are created)
        # We'll do this after buttons are created in _create_toolbar_button
        self._toolbar_button_wrappers = []
    
    def _create_toolbar_button(self, parent, icon, tooltip_text, command):
        """Create a small icon-only button"""
        btn = ctk.CTkButton(
            parent,
            text=icon,
            command=command,
            width=36,
            height=36,
            font=ctk.CTkFont(size=16),
            corner_radius=6,
            fg_color="transparent",  # Transparent background
            hover_color=("#3a3a3a", "#3a3a3a"),  # Slight hover effect
            border_width=0  # No border
        )
        btn.pack(side="left", padx=2, pady=5)
        
        # Bind drag handlers to button
        def button_press_handler(event):
            self.toolbar_dragging = False
            self.toolbar_drag_start_x = event.x_root
            self.toolbar_drag_start_y = event.y_root
            
            # Get current toolbar position - ensure using absolute positioning
            if self.floating_toolbar.winfo_exists():
                self.floating_toolbar.update_idletasks()
                self.toolbar_start_x = self.floating_toolbar.winfo_x()
                self.toolbar_start_y = self.floating_toolbar.winfo_y()
                
                # If position is 0 or invalid, try to get from place_info
                if self.toolbar_start_x == 0 and self.toolbar_start_y == 0:
                    try:
                        place_info = self.floating_toolbar.place_info()
                        if 'relx' in place_info and place_info['relx']:
                            # Convert relative to absolute
                            container_width = self.canvas_container.winfo_width()
                            container_height = self.canvas_container.winfo_height()
                            relx = float(place_info.get('relx', 0))
                            rely = float(place_info.get('rely', 0))
                            x_offset = int(place_info.get('x', 0))
                            y_offset = int(place_info.get('y', 0))
                            
                            # Calculate absolute position
                            self.toolbar_start_x = int(container_width * relx) + x_offset
                            self.toolbar_start_y = int(container_height * rely) + y_offset
                            
                            # Switch to absolute positioning
                            self.floating_toolbar.place(x=self.toolbar_start_x, y=self.toolbar_start_y)
                    except:
                        pass
            
            self.toolbar_pressed_widget = event.widget
            # Change cursor to move cursor (only on toolbar)
            self.floating_toolbar.configure(cursor="fleur")
        
        def button_motion_handler(event):
            dx = abs(event.x_root - self.toolbar_drag_start_x)
            dy = abs(event.y_root - self.toolbar_drag_start_y)
            
            if dx > 5 or dy > 5:
                if not self.toolbar_dragging:
                    # Just started dragging - ensure cursor is set
                    self.toolbar_dragging = True
                    self.floating_toolbar.configure(cursor="fleur")
                    self.canvas_container.configure(cursor="fleur")
            
            if self.toolbar_dragging:
                dx = event.x_root - self.toolbar_drag_start_x
                dy = event.y_root - self.toolbar_drag_start_y
                
                new_x = self.toolbar_start_x + dx
                new_y = self.toolbar_start_y + dy
                
                # Get container bounds without update_idletasks to reduce blinking
                container_width = self.canvas_container.winfo_width()
                container_height = self.canvas_container.winfo_height()
                
                # Get toolbar size without update_idletasks
                toolbar_width = self.floating_toolbar.winfo_reqwidth()
                toolbar_height = self.floating_toolbar.winfo_reqheight()
                
                new_x = max(0, min(new_x, container_width - toolbar_width))
                new_y = max(0, min(new_y, container_height - toolbar_height))
                
                # Update position directly without place_forget to avoid blinking
                # Ensure toolbar remains visible and on top
                if self.floating_toolbar.winfo_exists():
                    self.floating_toolbar.place(x=new_x, y=new_y)
                    self.floating_toolbar.lift()  # Ensure it's on top
        
        def button_release_handler(event):
            was_dragging = self.toolbar_dragging
            self.toolbar_dragging = False
            
            # Restore normal cursor (only on toolbar)
            self.floating_toolbar.configure(cursor="")
            
            if not was_dragging and self.toolbar_pressed_widget == btn:
                if command:
                    self.root.after(10, command)
            
            self.toolbar_pressed_widget = None
        
        btn.bind("<Button-1>", button_press_handler)
        btn.bind("<B1-Motion>", button_motion_handler)
        btn.bind("<ButtonRelease-1>", button_release_handler)
        
        return btn
    
    def _set_region_type(self, region_type: str):
        """Set the current region type for drawing"""
        if hasattr(self, 'region_canvas') and self.region_canvas:
            self.region_canvas.set_region_type(region_type)
            logger.debug(f"Region type set to: {region_type}")
    
    def _add_custom_region_type(self, region_type: str, label: str, color: str = "#ff00ff"):
        """Add a custom region type to the list"""
        if not hasattr(self, 'region_types_list'):
            self.region_types_list = []
        
        # Check if region type already exists
        existing_types = [rt[0] for rt in self.region_types_list]
        if region_type in existing_types:
            logger.warning(f"Region type '{region_type}' already exists")
            return False
        
        # Add to list
        self.region_types_list.append((region_type, label))
        
        # Add to canvas color mapping if not exists
        if hasattr(self, 'region_canvas') and self.region_canvas:
            if region_type not in self.region_canvas.region_colors:
                self.region_canvas.region_colors[region_type] = color
        
        # Add radio button with tick to scrollable frame
        if hasattr(self, 'region_type_scroll'):
            # Create a frame to hold radio button and tick mark
            item_frame = ctk.CTkFrame(self.region_type_scroll)
            item_frame.pack(fill="x", padx=5, pady=2)
            
            rb = ctk.CTkRadioButton(
                item_frame,
                text=label,
                variable=self.region_type_var,
                value=region_type,
                command=lambda rt=region_type: self._set_region_type(rt)
            )
            rb.pack(side="left", padx=5)
            
            # Create tick label (initially empty)
            tick_label = ctk.CTkLabel(
                item_frame,
                text="",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#00ff00"  # Green color
            )
            tick_label.pack(side="left", padx=5)
            
            # Store references
            if not hasattr(self, 'region_type_widgets'):
                self.region_type_widgets = {}
            self.region_type_widgets[region_type] = {
                'radio': rb,
                'tick': tick_label,
                'frame': item_frame
            }
            
            # Update ticks to check if this new type has saved regions
            self._update_region_type_ticks()
        
        logger.info(f"Added custom region type: {region_type} ({label})")
        return True
    
    def _add_custom_region_dialog(self):
        """Show dialog to add a custom region type"""
        # Create dialog window
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Add Custom Region Type")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Make dialog modal
        dialog.focus_set()
        
        # Region type name
        name_label = ctk.CTkLabel(dialog, text="Region Type Name (ID):", font=ctk.CTkFont(size=12, weight="bold"))
        name_label.pack(pady=(20, 5))
        
        name_entry = ctk.CTkEntry(dialog, width=300, placeholder_text="e.g., custom_button, npc_dialog, etc.")
        name_entry.pack(pady=5)
        
        # Display label
        label_label = ctk.CTkLabel(dialog, text="Display Label:", font=ctk.CTkFont(size=12, weight="bold"))
        label_label.pack(pady=(10, 5))
        
        label_entry = ctk.CTkEntry(dialog, width=300, placeholder_text="e.g., 🎯 Custom Button Region")
        label_entry.pack(pady=5)
        
        # Color picker (optional)
        color_label = ctk.CTkLabel(dialog, text="Color (hex, optional):", font=ctk.CTkFont(size=12, weight="bold"))
        color_label.pack(pady=(10, 5))
        
        color_entry = ctk.CTkEntry(dialog, width=300, placeholder_text="#ff00ff")
        color_entry.pack(pady=5)
        
        # Buttons
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(pady=20)
        
        def add_region():
            region_type = name_entry.get().strip()
            label = label_entry.get().strip()
            color = color_entry.get().strip() or "#ff00ff"
            
            if not region_type:
                messagebox.showwarning("Invalid Input", "Region type name is required!")
                return
            
            if not label:
                label = region_type.replace('_', ' ').title()
            
            # Validate color format
            if not color.startswith('#'):
                color = '#' + color
            if len(color) != 7:
                messagebox.showwarning("Invalid Color", "Color must be in hex format (e.g., #ff00ff)")
                return
            
            # Add the custom region
            if self._add_custom_region_type(region_type, label, color):
                messagebox.showinfo("Success", f"Custom region '{label}' added successfully!")
                dialog.destroy()
            else:
                messagebox.showerror("Error", f"Failed to add region type '{region_type}'. It may already exist.")
        
        add_btn = ctk.CTkButton(button_frame, text="Add", command=add_region, width=100)
        add_btn.pack(side="left", padx=10)
        
        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy, width=100)
        cancel_btn.pack(side="left", padx=10)
        
        # Focus on name entry
        name_entry.focus()
    
    def _take_screenshot_for_region(self):
        """Take screenshot from selected device for region configuration"""
        device_id = self.region_device_var.get()
        if device_id == "Select device..." or not device_id:
            messagebox.showwarning("No Device", "Please select a device first.")
            return
        
        if device_id not in self.multi_device_manager.get_connected_devices():
            messagebox.showwarning("Device Not Connected", f"Device {device_id} is not connected.")
            return
        
        def screenshot_thread():
            try:
                self._update_status(f"📸 Taking screenshot from {device_id} for region configuration...")
                
                result = self.multi_device_manager.execute_on_device(
                    device_id,
                    'take_screenshot'
                )
                
                if result is not None:
                    # Update canvas on main thread
                    self.root.after(0, lambda: self.region_canvas.load_screenshot(result))
                    self._update_status(f"✅ Screenshot loaded for region configuration")
                else:
                    raise Exception("Failed to capture screenshot")
                    
            except Exception as e:
                error_msg = f"Error taking screenshot: {e}"
                logger.error(error_msg, exc_info=True)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        threading.Thread(target=screenshot_thread, daemon=True).start()
    
    def _clear_current_region(self):
        """Clear regions of the currently selected type"""
        if hasattr(self, 'region_canvas') and self.region_canvas:
            region_type = self.region_type_var.get()
            self.region_canvas.clear_region(region_type)
            self._update_status(f"Cleared {region_type} regions")
            # Update ticks after clearing
            self._update_region_type_ticks()
    
    def _clear_all_regions(self):
        """Clear all regions"""
        if hasattr(self, 'region_canvas') and self.region_canvas:
            if messagebox.askyesno("Clear All Regions", "Are you sure you want to clear all regions?"):
                self.region_canvas.clear_all_regions()
                self._update_status("Cleared all regions")
                # Update ticks after clearing
                self._update_region_type_ticks()
    
    def _save_regions(self):
        """Save regions to file"""
        if not hasattr(self, 'region_canvas') or not self.region_canvas:
            messagebox.showwarning("No Canvas", "Canvas not initialized.")
            return
        
        # Get device ID if available for per-device regions
        device_id = self.region_device_var.get() if hasattr(self, 'region_device_var') else None
        if device_id and device_id != "Select device...":
            # Use device-specific filename
            default_filename = f"regions_{device_id.replace(':', '_').replace('.', '_')}.json"
        else:
            default_filename = "regions.json"
        
        # Ensure config directory exists
        config_dir = Path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default path in config folder
        file_path = config_dir / default_filename
        
        # Show confirmation dialog with save path
        confirm_msg = f"Save regions to:\n{file_path}\n\nProceed?"
        if messagebox.askyesno("Confirm Save", confirm_msg):
            if self.region_canvas.save_regions(str(file_path)):
                messagebox.showinfo("Success", f"Regions saved to {file_path}")
                self._update_status(f"Regions saved to {file_path}")
                # Update ticks after saving
                self._update_region_type_ticks()
            else:
                messagebox.showerror("Error", "Failed to save regions")
    
    def _load_regions(self):
        """Load regions from file"""
        if not hasattr(self, 'region_canvas') or not self.region_canvas:
            messagebox.showwarning("No Canvas", "Canvas not initialized.")
            return
        
        # Ensure config directory exists
        config_dir = Path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(config_dir)
        )
        
        if file_path:
            if self.region_canvas.load_regions(file_path):
                messagebox.showinfo("Success", f"Regions loaded from {file_path}")
                self._update_status(f"Regions loaded from {file_path}")
                # Update ticks after loading
                self._update_region_type_ticks()
            else:
                messagebox.showerror("Error", "Failed to load regions")
    
    def _update_region_device_list(self):
        """Update device dropdown in region tab"""
        try:
            if not hasattr(self, 'region_device_dropdown') or not self.region_device_dropdown:
                return
            
            # Get connected devices
            connected_devices = list(self.multi_device_manager.get_connected_devices().keys())
            if connected_devices:
                # Update dropdown
                self.region_device_dropdown.configure(values=connected_devices)
                current_value = self.region_device_var.get()
                if not current_value or current_value == "Select device..." or current_value not in connected_devices:
                    if connected_devices:
                        self.region_device_var.set(connected_devices[0])
            else:
                # No devices connected
                self.region_device_dropdown.configure(values=["Select device..."])
                self.region_device_var.set("Select device...")
        except Exception as e:
            logger.debug(f"Error updating region device list: {e}")
        
        # Update ticks when device list changes
        self._update_region_type_ticks()
        # Auto-load regions when device is selected
        self._auto_load_regions()
    
    def _load_custom_region_types_from_files(self):
        """Load custom region types from JSON files and add them to the UI"""
        try:
            # Determine which JSON file to check
            device_id = None
            if hasattr(self, 'region_device_var'):
                device_id = self.region_device_var.get()
            
            if device_id and device_id != "Select device...":
                # Check device-specific file
                regions_file = Path("config") / f"regions_{device_id.replace(':', '_').replace('.', '_')}.json"
            else:
                # Check default file
                regions_file = Path("config/regions.json")
            
            # Also check default file if device-specific doesn't exist
            if not regions_file.exists():
                regions_file = Path("config/regions.json")
            
            # Load region types from JSON file
            if regions_file.exists():
                try:
                    with open(regions_file, 'r', encoding='utf-8') as f:
                        saved_regions = json.load(f)
                    
                    # Get existing region types
                    existing_types = [rt[0] for rt in self.region_types_list] if hasattr(self, 'region_types_list') else []
                    
                    # Add missing region types from JSON
                    for region_type in saved_regions.keys():
                        if region_type not in existing_types:
                            # Generate a label from the region type name
                            label = region_type.replace('_', ' ').title()
                            # Add emoji if it's a custom type
                            if region_type not in ['player', 'quests', 'control_buttons', 'items', 'health_bar', 'mp_bar', 'other']:
                                label = f"🔧 {label}"
                            
                            # Add to region types list
                            self.region_types_list.append((region_type, label))
                            
                            # Add radio button with tick if not already in widgets
                            if not hasattr(self, 'region_type_widgets') or region_type not in self.region_type_widgets:
                                # Create a frame to hold radio button and tick mark
                                item_frame = ctk.CTkFrame(self.region_type_scroll)
                                item_frame.pack(fill="x", padx=5, pady=2)
                                
                                rb = ctk.CTkRadioButton(
                                    item_frame,
                                    text=label,
                                    variable=self.region_type_var,
                                    value=region_type,
                                    command=lambda rt=region_type: self._set_region_type(rt)
                                )
                                rb.pack(side="left", padx=5)
                                
                                # Create tick label
                                tick_label = ctk.CTkLabel(
                                    item_frame,
                                    text="",
                                    font=ctk.CTkFont(size=14, weight="bold"),
                                    text_color="#00ff00"  # Green color
                                )
                                tick_label.pack(side="left", padx=5)
                                
                                # Store references
                                if not hasattr(self, 'region_type_widgets'):
                                    self.region_type_widgets = {}
                                self.region_type_widgets[region_type] = {
                                    'radio': rb,
                                    'tick': tick_label,
                                    'frame': item_frame
                                }
                            
                            # Add color to canvas if not exists
                            if hasattr(self, 'region_canvas') and self.region_canvas:
                                if region_type not in self.region_canvas.region_colors:
                                    # Assign a default color for custom types
                                    self.region_canvas.region_colors[region_type] = "#ff00ff"  # Magenta
                            
                            logger.info(f"Loaded custom region type from file: {region_type} ({label})")
                
                except Exception as e:
                    logger.debug(f"Error loading custom region types from file: {e}")
        
        except Exception as e:
            logger.debug(f"Error in _load_custom_region_types_from_files: {e}")
    
    def _auto_load_regions(self):
        """Automatically load saved regions when device is selected"""
        if not hasattr(self, 'region_canvas') or not self.region_canvas:
            return
        
        try:
            # Determine which JSON file to load
            device_id = None
            if hasattr(self, 'region_device_var'):
                device_id = self.region_device_var.get()
            
            if device_id and device_id != "Select device...":
                # Load device-specific file
                regions_file = Path("config") / f"regions_{device_id.replace(':', '_').replace('.', '_')}.json"
            else:
                # Load default file
                regions_file = Path("config/regions.json")
            
            # Load regions if file exists
            if regions_file.exists():
                if self.region_canvas.load_regions(str(regions_file)):
                    logger.info(f"Auto-loaded regions from {regions_file}")
                else:
                    logger.debug(f"Failed to auto-load regions from {regions_file}")
        
        except Exception as e:
            logger.debug(f"Error auto-loading regions: {e}")
    
    def _update_region_type_ticks(self):
        """Update green tick marks for region types that have saved regions"""
        if not hasattr(self, 'region_type_widgets'):
            return
        
        try:
            # Determine which JSON file to check
            device_id = None
            if hasattr(self, 'region_device_var'):
                device_id = self.region_device_var.get()
            
            if device_id and device_id != "Select device...":
                # Check device-specific file
                regions_file = Path("config") / f"regions_{device_id.replace(':', '_').replace('.', '_')}.json"
            else:
                # Check default file
                regions_file = Path("config/regions.json")
            
            # Load regions if file exists
            saved_regions = {}
            if regions_file.exists():
                try:
                    with open(regions_file, 'r', encoding='utf-8') as f:
                        saved_regions = json.load(f)
                except Exception as e:
                    logger.debug(f"Error reading regions file: {e}")
            
            # Update tick marks for each region type
            for region_type, widgets in self.region_type_widgets.items():
                tick_label = widgets['tick']
                if region_type in saved_regions and saved_regions[region_type]:
                    # Has regions - show green tick
                    tick_label.configure(text="✓")
                else:
                    # No regions - clear tick
                    tick_label.configure(text="")
        
        except Exception as e:
            logger.debug(f"Error updating region type ticks: {e}")
    
    def _load_config_display(self):
        """Load and display current configuration"""
        try:
            import yaml
            config_file = Path("config/bot_config.yaml")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_text = f.read()
                self.config_text.delete("1.0", "end")
                self.config_text.insert("1.0", config_text)
            else:
                self.config_text.insert("1.0", "# Configuration file not found")
        except Exception as e:
            logger.error(f"Error loading config display: {e}")
            self.config_text.insert("1.0", f"Error loading configuration: {e}")
    
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
                    
                    # Process screenshot queue (limit to 1 at a time to prevent memory buildup)
                    screenshot = None
                    try:
                        # Get only the most recent screenshot, discard old ones
                        while not self.screenshot_queue.empty():
                            if screenshot is not None:
                                del screenshot  # Release old screenshot
                            screenshot = self.screenshot_queue.get_nowait()
                        
                        if screenshot is not None:
                            self._update_screenshot_display(screenshot)
                            del screenshot  # Release after display
                    except Exception as e:
                        logger.debug(f"Error processing screenshot queue: {e}")
                    
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