"""
Region Configuration Module - Canvas-based region selection for game UI elements
Allows drawing rectangles on screenshots to define regions (player, quests, controls, etc.)
"""

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import config_manager

logger = get_logger(__name__)


class RegionCanvas:
    """Canvas widget for displaying screenshots and drawing regions"""
    
    def __init__(self, parent, width=800, height=600):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Create frame for canvas with scrollbars
        self.frame = ctk.CTkFrame(parent)
        
        # Create canvas (using tkinter Canvas for better drawing support)
        self.canvas = tk.Canvas(
            self.frame,
            width=width,
            height=height,
            bg='#2b2b2b',
            highlightthickness=1,
            highlightbackground='#565b5e'
        )
        
        # Scrollbars
        self.v_scrollbar = tk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = tk.Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Pack scrollbars and canvas
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)
        
        # Current screenshot (store as numpy array, convert to PIL only when needed)
        self.current_screenshot = None
        self.original_screenshot = None  # Store original as numpy array (more efficient)
        self.screenshot_image = None  # PhotoImage for display
        self.canvas_image_id = None
        self.original_pil_image = None  # Lazy-loaded PIL image for zooming
        
        # Zoom state
        self.zoom_level = 1.0  # Current zoom level (1.0 = 100%)
        self.min_zoom = 0.1    # Minimum zoom (10%)
        self.max_zoom = 5.0    # Maximum zoom (500%)
        self.zoom_step = 0.1   # Zoom step size (10% increments)
        
        # Drawing state
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        
        # Regions dictionary: {region_type: [(x1, y1, x2, y2), ...]}
        self.regions: Dict[str, List[Tuple[int, int, int, int]]] = {}
        
        # Current region type being drawn
        self.current_region_type = None
        
        # Flag to control whether to clear previous regions when drawing new one
        self.clear_previous_on_draw = True
        
        # Region colors
        self.region_colors = {
            'player': '#00ff00',      # Green
            'quests': '#0080ff',      # Blue
            'control_buttons': '#ff8000',  # Orange
            'items': '#ffff00',       # Yellow
            'health_bar': '#ff0000',  # Red
            'mp_bar': '#0000ff',      # Dark Blue
            'other': '#ff00ff'        # Magenta
        }
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)
        
        # Bind mouse wheel for zooming (Windows/Linux)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        # Bind mouse wheel for zooming (Mac)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        
        # Make canvas focusable for mouse wheel events
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<Leave>", lambda e: self.canvas.focus_set())
        
    def pack(self, **kwargs):
        """Pack the canvas frame"""
        self.frame.pack(**kwargs)
    
    def load_screenshot(self, screenshot: np.ndarray):
        """Load a screenshot (numpy array) onto the canvas"""
        try:
            # Release old screenshots and images if exist (before storing new ones)
            if self.current_screenshot is not None:
                del self.current_screenshot
            if self.original_screenshot is not None:
                del self.original_screenshot
            if self.original_pil_image is not None:
                del self.original_pil_image
            if self.screenshot_image is not None:
                # PhotoImage will be released when canvas deletes the image
                self.screenshot_image = None
            
            # Convert BGR to RGB for display (PIL/Tkinter expects RGB)
            # OpenCV images are in BGR format, but we need RGB for display
            # take_screenshot() should always return BGR format, so we convert BGR -> RGB
            if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
                # Convert BGR to RGB for display
                try:
                    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2RGB'):
                        rgb_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                        logger.debug("Converted BGR to RGB using cv2.cvtColor")
                    else:
                        # Manual BGR to RGB conversion using numpy (reverse channel order)
                        rgb_image = screenshot[:, :, ::-1]  # Reverse channels: BGR -> RGB
                        logger.debug("Converted BGR to RGB using manual numpy conversion")
                except Exception as e:
                    logger.warning(f"Error converting BGR to RGB with cv2, trying manual conversion: {e}")
                    # Fallback: manual conversion (reverse channel order)
                    rgb_image = screenshot[:, :, ::-1]  # Reverse channels: BGR -> RGB
                    logger.debug("Used fallback manual BGR to RGB conversion")
            elif len(screenshot.shape) == 3 and screenshot.shape[2] == 4:
                # RGBA image - convert to RGB
                try:
                    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGRA2RGB'):
                        rgb_image = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
                    elif hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_RGBA2RGB'):
                        # If already RGBA, just drop alpha
                        rgb_image = screenshot[:, :, :3]
                    else:
                        # Manual: drop alpha and reverse if BGRA
                        rgb_image = screenshot[:, :, :3][:, :, ::-1]  # Drop alpha, reverse BGR->RGB
                except Exception as e:
                    logger.warning(f"Error converting BGRA to RGB: {e}")
                    rgb_image = screenshot[:, :, :3][:, :, ::-1]  # Drop alpha, reverse BGR->RGB
            else:
                # Grayscale or other format
                rgb_image = screenshot
            
            # Store original numpy array (more memory efficient than PIL)
            # We'll convert to PIL only when needed for display/zooming
            self.original_screenshot = rgb_image.copy() if rgb_image is not None else None
            
            # Store original size
            self.original_height, self.original_width = rgb_image.shape[:2]
            
            # Don't convert to PIL immediately - do it lazily when needed for display
            # This saves memory if screenshot is loaded but not displayed
            self.original_pil_image = None
            
            # Reset zoom level when loading new screenshot
            self.zoom_level = 1.0
            
            # Calculate initial scale to fit canvas while maintaining aspect ratio
            scale_w = self.width / self.original_width
            scale_h = self.height / self.original_height
            self.base_scale = min(scale_w, scale_h, 1.0)  # Don't scale up beyond original
            
            # Apply zoom to base scale
            self.scale = self.base_scale * self.zoom_level
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Store screenshot for reference (numpy array, more efficient)
            # Note: We keep the original numpy array, not the RGB copy
            self.current_screenshot = screenshot
            
            # Update image display with current zoom
            self._update_image_display()
            
            # Redraw regions
            self._redraw_regions()
            
            logger.info(f"Screenshot loaded: {self.original_width}x{self.original_height} (zoom: {self.zoom_level:.1%}, scale: {self.scale:.3f})")
            
        except Exception as e:
            logger.error(f"Error loading screenshot: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load screenshot: {e}")
    
    def set_region_type(self, region_type: str):
        """Set the current region type for drawing"""
        self.current_region_type = region_type
        logger.debug(f"Region type set to: {region_type}")
    
    def _on_button_press(self, event):
        """Handle mouse button press - start drawing rectangle"""
        if not self.current_region_type:
            return
        
        # Clear previous regions of the same type if configured to do so
        if self.clear_previous_on_draw and self.current_region_type in self.regions:
            logger.debug(f"Clearing previous {len(self.regions[self.current_region_type])} region(s) of type '{self.current_region_type}' before drawing new one")
            del self.regions[self.current_region_type]
            # Redraw to remove the cleared regions
            self._redraw_regions()
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        self.start_x = canvas_x
        self.start_y = canvas_y
        self.drawing = True
        
        # Create rectangle
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=self.region_colors.get(self.current_region_type, '#ffffff'),
            width=2,
            tags="drawing"
        )
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag - update rectangle"""
        if not self.drawing or self.current_rect is None:
            return
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Update rectangle
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, canvas_x, canvas_y)
    
    def _on_button_release(self, event):
        """Handle mouse button release - finish drawing rectangle"""
        if not self.drawing or self.current_rect is None:
            return
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Calculate rectangle coordinates
        x1 = min(self.start_x, canvas_x)
        y1 = min(self.start_y, canvas_y)
        x2 = max(self.start_x, canvas_x)
        y2 = max(self.start_y, canvas_y)
        
        # Convert to original image coordinates
        orig_x1 = int(x1 / self.scale) if hasattr(self, 'scale') else int(x1)
        orig_y1 = int(y1 / self.scale) if hasattr(self, 'scale') else int(y1)
        orig_x2 = int(x2 / self.scale) if hasattr(self, 'scale') else int(x2)
        orig_y2 = int(y2 / self.scale) if hasattr(self, 'scale') else int(y2)
        
        # Ensure valid coordinates
        if abs(orig_x2 - orig_x1) > 5 and abs(orig_y2 - orig_y1) > 5:
            # Add region
            if self.current_region_type not in self.regions:
                self.regions[self.current_region_type] = []
            
            self.regions[self.current_region_type].append((orig_x1, orig_y1, orig_x2, orig_y2))
            logger.info(f"Region added: {self.current_region_type} = ({orig_x1}, {orig_y1}, {orig_x2}, {orig_y2})")
        
        # Remove temporary rectangle
        self.canvas.delete(self.current_rect)
        self.current_rect = None
        self.drawing = False
        
        # Redraw all regions
        self._redraw_regions()
    
    def _redraw_regions(self):
        """Redraw all regions on the canvas"""
        # Clear existing region rectangles
        self.canvas.delete("region")
        
        if not hasattr(self, 'scale'):
            return
        
        # Draw each region
        for region_type, region_list in self.regions.items():
            color = self.region_colors.get(region_type, '#ffffff')
            for x1, y1, x2, y2 in region_list:
                # Scale coordinates for display
                scaled_x1 = x1 * self.scale
                scaled_y1 = y1 * self.scale
                scaled_x2 = x2 * self.scale
                scaled_y2 = y2 * self.scale
                
                # Draw rectangle
                self.canvas.create_rectangle(
                    scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                    outline=color,
                    width=2,
                    tags="region"
                )
                
                # Draw label
                label_text = region_type.replace('_', ' ').title()
                self.canvas.create_text(
                    scaled_x1 + 5, scaled_y1 + 5,
                    text=label_text,
                    fill=color,
                    anchor="nw",
                    font=("Arial", 10, "bold"),
                    tags="region"
                )
    
    def clear_region(self, region_type: str):
        """Clear all regions of a specific type"""
        if region_type in self.regions:
            del self.regions[region_type]
            self._redraw_regions()
            logger.info(f"Cleared regions for: {region_type}")
    
    def clear_all_regions(self):
        """Clear all regions"""
        self.regions = {}
        self._redraw_regions()
        logger.info("Cleared all regions")
    
    def get_regions(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Get all regions"""
        return self.regions.copy()
    
    def set_regions(self, regions: Dict[str, List[Tuple[int, int, int, int]]]):
        """Set regions from dictionary"""
        self.regions = regions.copy()
        self._redraw_regions()
        logger.info(f"Loaded {sum(len(v) for v in regions.values())} regions")
    
    def save_regions(self, file_path: str):
        """Save regions to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.regions, f, indent=2)
            logger.info(f"Regions saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving regions: {e}")
            return False
    
    def load_regions(self, file_path: str):
        """Load regions from JSON file"""
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.regions = json.load(f)
                # Convert lists to tuples
                for region_type in self.regions:
                    self.regions[region_type] = [tuple(r) for r in self.regions[region_type]]
                self._redraw_regions()
                logger.info(f"Regions loaded from {file_path}")
                return True
            else:
                logger.warning(f"Regions file not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading regions: {e}")
            return False
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling for zooming"""
        if self.original_pil_image is None:
            return
        
        # Determine zoom direction
        # Windows/Linux: event.delta > 0 is scroll up (zoom in), < 0 is scroll down (zoom out)
        # Mac: event.num == 4 is scroll up, event.num == 5 is scroll down
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            # Zoom in
            new_zoom = min(self.zoom_level + self.zoom_step, self.max_zoom)
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            # Zoom out
            new_zoom = max(self.zoom_level - self.zoom_step, self.min_zoom)
        else:
            return
        
        # Get mouse position relative to canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Zoom centered on mouse position
        self._zoom_to(new_zoom, canvas_x, canvas_y)
    
    def _zoom_to(self, zoom_level: float, center_x: float = None, center_y: float = None):
        """Zoom to a specific level, optionally centered on a point"""
        if self.original_pil_image is None:
            return
        
        old_zoom = self.zoom_level
        self.zoom_level = max(self.min_zoom, min(zoom_level, self.max_zoom))
        
        if self.zoom_level == old_zoom:
            return  # No change
        
        # If no center point specified, use canvas center
        if center_x is None or center_y is None:
            center_x = self.canvas.winfo_width() / 2
            center_y = self.canvas.winfo_height() / 2
        
        # Get current scroll position (visible area in canvas coordinates)
        scroll_x = self.canvas.canvasx(0)
        scroll_y = self.canvas.canvasy(0)
        
        # Calculate the point under the mouse in original image coordinates
        # Using the old scale to get the original coordinates
        old_scale = self.base_scale * old_zoom
        orig_center_x = (scroll_x + center_x) / old_scale
        orig_center_y = (scroll_y + center_y) / old_scale
        
        # Update scale
        self.scale = self.base_scale * self.zoom_level
        
        # Update image display
        self._update_image_display()
        
        # Calculate new scroll position to keep the same point under the mouse
        # Using the new scale to calculate new scroll position
        new_scroll_x = orig_center_x * self.scale - center_x
        new_scroll_y = orig_center_y * self.scale - center_y
        
        # Get the new image size
        new_image_width = int(self.original_width * self.scale)
        new_image_height = int(self.original_height * self.scale)
        
        # Clamp scroll position to valid range
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        max_scroll_x = max(0, new_image_width - canvas_width)
        max_scroll_y = max(0, new_image_height - canvas_height)
        
        new_scroll_x = max(0, min(new_scroll_x, max_scroll_x))
        new_scroll_y = max(0, min(new_scroll_y, max_scroll_y))
        
        # Update scroll position (convert to scroll fraction)
        if new_image_width > canvas_width:
            scroll_x_fraction = new_scroll_x / (new_image_width - canvas_width)
        else:
            scroll_x_fraction = 0
        
        if new_image_height > canvas_height:
            scroll_y_fraction = new_scroll_y / (new_image_height - canvas_height)
        else:
            scroll_y_fraction = 0
        
        self.canvas.xview_moveto(scroll_x_fraction)
        self.canvas.yview_moveto(scroll_y_fraction)
        
        # Redraw regions
        self._redraw_regions()
        
        logger.debug(f"Zoomed to {self.zoom_level:.1%} (scale: {self.scale:.3f})")
    
    def _update_image_display(self):
        """Update the displayed image with current zoom level"""
        if self.original_screenshot is None:
            return
        
        # Calculate new size with current zoom
        new_width = int(self.original_width * self.scale)
        new_height = int(self.original_height * self.scale)
        
        # Convert numpy array to PIL only when needed (more memory efficient)
        if self.original_pil_image is None and self.original_screenshot is not None:
            self.original_pil_image = Image.fromarray(self.original_screenshot)
        
        # Resize image
        resized_image = self.original_pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Release old PhotoImage if exists
        if hasattr(self, 'screenshot_image') and self.screenshot_image is not None:
            # PhotoImage will be garbage collected when canvas no longer references it
            pass
        
        # Convert to PhotoImage
        self.screenshot_image = ImageTk.PhotoImage(resized_image)
        
        # Release resized PIL image immediately after PhotoImage creation
        del resized_image
        
        # Update canvas image
        if self.canvas_image_id:
            self.canvas.delete(self.canvas_image_id)
        
        # Add image to canvas
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.screenshot_image)
        
        # Update canvas scroll region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def zoom_in(self):
        """Zoom in"""
        if self.original_pil_image is None:
            return
        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        self._zoom_to(self.zoom_level + self.zoom_step, canvas_center_x, canvas_center_y)
    
    def zoom_out(self):
        """Zoom out"""
        if self.original_pil_image is None:
            return
        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        self._zoom_to(self.zoom_level - self.zoom_step, canvas_center_x, canvas_center_y)
    
    def zoom_fit(self):
        """Reset zoom to fit canvas"""
        if self.original_pil_image is None:
            return
        self.zoom_level = 1.0
        self.scale = self.base_scale
        self._update_image_display()
        self._redraw_regions()
        # Center the image
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        logger.info(f"Zoom reset to fit (scale: {self.scale:.3f})")

