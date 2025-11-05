"""
Memory monitoring utility for real-time RAM usage tracking with detailed component analysis
"""

import psutil
import os
import gc
import sys
import tracemalloc
import warnings
from typing import Dict, Optional, List, Tuple
from collections import deque, defaultdict
import threading
import time
import inspect

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemoryMonitor:
    """
    Real-time memory usage monitor for the bot application
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize memory monitor
        
        Args:
            max_history: Maximum number of historical data points to keep
        """
        self.process = psutil.Process(os.getpid())
        self.max_history = max_history
        self.monitoring = False
        self.monitor_thread = None
        self.update_callback = None
        
        # Historical data
        self.memory_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        
        # Memory breakdown tracking
        self.component_memory = {}
        
        # Detailed tracking
        self.memory_snapshots = []
        self.component_tracking = defaultdict(list)
        self.tracemalloc_enabled = False
        
        # Track GC collection rate
        self.last_gc_collections = 0
        self.last_gc_check_time = 0.0  # Initialize to 0
        self.gc_collection_rate = 0.0  # Collections per second
        
        # Cache object lists to avoid calling gc.get_objects() too frequently
        self._cached_objects = None
        self._cache_timestamp = 0
        self._cache_ttl = 5.0  # Cache for 5 seconds
        
        # Start tracemalloc for detailed tracking
        try:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            logger.info("Tracemalloc enabled for detailed memory tracking")
        except Exception as e:
            logger.warning(f"Could not enable tracemalloc: {e}")
        
    def get_memory_info(self) -> Dict:
        """
        Get current memory usage information
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            # Process memory
            process_info = self.process.memory_info()
            process_memory_mb = process_info.rss / 1024 / 1024  # RSS in MB
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_total_mb = system_memory.total / 1024 / 1024
            system_available_mb = system_memory.available / 1024 / 1024
            system_used_mb = system_memory.used / 1024 / 1024
            system_percent = system_memory.percent
            
            # Memory breakdown by component
            breakdown = self._get_memory_breakdown()
            
            return {
                'process_memory_mb': round(process_memory_mb, 2),
                'process_memory_gb': round(process_memory_mb / 1024, 2),
                'system_total_mb': round(system_total_mb, 2),
                'system_total_gb': round(system_total_mb / 1024, 2),
                'system_available_mb': round(system_available_mb, 2),
                'system_used_mb': round(system_used_mb, 2),
                'system_percent': round(system_percent, 2),
                'breakdown': breakdown,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {
                'process_memory_mb': 0,
                'process_memory_gb': 0,
                'system_total_mb': 0,
                'system_available_mb': 0,
                'system_used_mb': 0,
                'system_percent': 0,
                'breakdown': {},
                'timestamp': time.time()
            }
    
    def _get_memory_breakdown(self) -> Dict:
        """
        Get detailed memory breakdown by component
        
        Returns:
            Dictionary with detailed memory usage by component
        """
        breakdown = {}
        
        try:
            # Get garbage collection stats (more detailed)
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats)
            
            # Calculate GC collection rate (collections per second)
            current_time = time.time()
            if self.last_gc_check_time > 0:
                time_diff = current_time - self.last_gc_check_time
                if time_diff > 0:
                    collections_diff = gc_collections - self.last_gc_collections
                    self.gc_collection_rate = collections_diff / time_diff
            self.last_gc_collections = gc_collections
            self.last_gc_check_time = current_time
            
            # Calculate GC statistics
            gc_details = {
                'total_collections': gc_collections,
                'collections_per_second': round(self.gc_collection_rate, 2),
                'collections_by_generation': {}
            }
            
            # Get detailed stats per generation
            for i, stat in enumerate(gc_stats):
                generation_name = ['Generation 0', 'Generation 1', 'Generation 2'][i] if i < 3 else f'Generation {i}'
                gc_details['collections_by_generation'][generation_name] = {
                    'collections': stat['collections'],
                    'collected': stat.get('collected', 0),
                    'uncollectable': stat.get('uncollectable', 0)
                }
            
            # Get GC threshold (when GC will trigger)
            gc_threshold = gc.get_threshold()
            gc_details['threshold'] = gc_threshold
            
            # Get current counts per generation
            gc_counts = gc.get_count()
            gc_details['current_counts'] = {
                'gen0': gc_counts[0],
                'gen1': gc_counts[1],
                'gen2': gc_counts[2]
            }
            
            # Check if GC is running too frequently (potential memory pressure)
            if self.gc_collection_rate > 10.0:  # More than 10 collections per second
                gc_details['warning'] = f"High GC rate: {self.gc_collection_rate:.1f} collections/sec (possible memory pressure)"
            
            # Count objects by type
            object_counts = {}
            object_sizes = {}
            
            # Detailed component tracking
            component_details = {}
            
            import numpy as np
            import cv2
            
            # Check for numpy arrays (screenshots) - this is often the biggest memory consumer
            # Use a more efficient approach to avoid iterating all objects multiple times
            numpy_arrays = []
            numpy_memory = 0
            large_arrays_list = []
            
            # Get all objects once (use cache to avoid expensive calls)
            # gc.get_objects() is expensive and can trigger GC, so we cache it
            current_cache_time = time.time()
            if (self._cached_objects is None or 
                (current_cache_time - self._cache_timestamp) > self._cache_ttl):
                try:
                    self._cached_objects = gc.get_objects()
                    self._cache_timestamp = current_cache_time
                except Exception as e:
                    logger.debug(f"Error getting objects: {e}")
                    # Use empty list if we can't get objects
                    self._cached_objects = []
            
            all_objects = self._cached_objects if self._cached_objects else []
            
            # Limit iteration to avoid performance issues
            max_objects_to_check = 100000  # Limit to first 100k objects
            objects_to_check = all_objects[:max_objects_to_check] if len(all_objects) > max_objects_to_check else all_objects
            
            # Suppress PyTorch deprecation warnings when checking object types
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.distributed.reduce_op.*')
                
                for obj in objects_to_check:
                    if isinstance(obj, np.ndarray) and hasattr(obj, 'nbytes'):
                        try:
                            numpy_arrays.append(obj)
                            size_mb = obj.nbytes / 1024 / 1024
                            numpy_memory += size_mb
                            # Track large arrays
                            if size_mb > 1.0:  # Arrays larger than 1MB
                                large_arrays_list.append({
                                    'size_mb': round(size_mb, 2),
                                    'shape': obj.shape,
                                    'dtype': str(obj.dtype)
                                })
                        except (AttributeError, TypeError):
                            # Skip invalid arrays
                            pass
            
            if large_arrays_list:
                component_details['large_numpy_arrays'] = large_arrays_list
            
            # Check for PIL images (use same object list)
            from PIL import Image
            pil_images = []
            pil_memory = 0
            # Continue suppressing warnings for PIL image check
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.distributed.reduce_op.*')
                
                for obj in objects_to_check:
                    if isinstance(obj, Image.Image):
                        pil_images.append(obj)
                        try:
                            size_mb = len(obj.tobytes()) / 1024 / 1024
                            pil_memory += size_mb
                        except:
                            pass
            
            # Count all objects by type (use same object list to avoid multiple iterations)
            for obj in objects_to_check:
                try:
                    obj_type = type(obj).__name__
                    
                    if obj_type not in object_counts:
                        object_counts[obj_type] = 0
                        object_sizes[obj_type] = 0
                    object_counts[obj_type] += 1
                    
                    # Estimate size (rough)
                    try:
                        obj_sz = sys.getsizeof(obj)
                        object_sizes[obj_type] += obj_sz
                    except:
                        pass
                except:
                    # Skip objects that can't be inspected
                    pass
            
            # Check for EasyOCR reader (if loaded)
            easyocr_memory = 0
            try:
                from ..utils.ocr_reader import shared_ocr_reader
                if shared_ocr_reader._reader is not None:
                    # EasyOCR models are typically 1-2GB
                    # We can't easily measure exact size, but we can detect if it's loaded
                    easyocr_memory = 1500  # Estimated 1.5GB when loaded
                    component_details['easyocr_loaded'] = True
                else:
                    component_details['easyocr_loaded'] = False
            except:
                component_details['easyocr_loaded'] = False
            
            # Get tracemalloc statistics if enabled
            tracemalloc_stats = {}
            if self.tracemalloc_enabled:
                try:
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    
                    # Get top memory consumers
                    total_traced = sum(stat.size for stat in top_stats)
                    tracemalloc_stats = {
                        'total_traced_mb': round(total_traced / 1024 / 1024, 2),
                        'top_consumers': []
                    }
                    
                    # Top 10 memory consumers
                    for stat in top_stats[:10]:
                        size_mb = stat.size / 1024 / 1024
                        if size_mb > 0.1:  # Only show significant consumers (>100KB)
                            tracemalloc_stats['top_consumers'].append({
                                'file': stat.traceback[0].filename if stat.traceback else 'unknown',
                                'line': stat.traceback[0].lineno if stat.traceback else 0,
                                'size_mb': round(size_mb, 2),
                                'count': stat.count
                            })
                except Exception as e:
                    logger.debug(f"Error getting tracemalloc stats: {e}")
            
            # Calculate object sizes (top 10 by total size)
            top_object_sizes = sorted(
                [(obj_type, object_sizes[obj_type] / 1024 / 1024) for obj_type in object_sizes.keys()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            breakdown = {
                'numpy_arrays_mb': round(numpy_memory, 2),
                'numpy_array_count': len(numpy_arrays),
                'pil_images_mb': round(pil_memory, 2),
                'pil_image_count': len(pil_images),
                'easyocr_mb': round(easyocr_memory, 2),
                'gc_collections': gc_collections,
                'gc_details': gc_details,
                'total_objects': len(object_counts),
                'top_object_types_by_count': dict(sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'top_object_types_by_size': {k: round(v, 2) for k, v in top_object_sizes},
                'component_details': component_details,
                'tracemalloc': tracemalloc_stats
            }
            
            # Find large arrays (potential memory leaks)
            if 'large_numpy_arrays' in component_details:
                large_arrays = component_details['large_numpy_arrays']
                if len(large_arrays) > 5:  # More than 5 large arrays might indicate a leak
                    breakdown['potential_leak'] = f"Found {len(large_arrays)} large numpy arrays (>1MB each)"
            
        except Exception as e:
            logger.error(f"Error getting memory breakdown: {e}", exc_info=True)
            breakdown = {
                'error': str(e),
                'numpy_arrays_mb': 0,
                'numpy_array_count': 0,
                'pil_images_mb': 0,
                'pil_image_count': 0,
                'easyocr_mb': 0,
                'gc_collections': 0,
                'total_objects': 0,
                'component_details': {'easyocr_loaded': False},
                'tracemalloc': {}
            }
        
        return breakdown
    
    def start_monitoring(self, interval: float = 2.0, callback: Optional[callable] = None):
        """
        Start real-time memory monitoring
        
        Args:
            interval: Update interval in seconds (default: 2.0 to reduce GC pressure)
            callback: Optional callback function to call with memory info
        """
        if self.monitoring:
            logger.warning("Memory monitoring already started")
            return
        
        self.monitoring = True
        self.update_callback = callback
        
        # Increase cache TTL based on interval to reduce GC.get_objects() calls
        self._cache_ttl = max(interval * 2, 5.0)  # Cache for at least 2x interval or 5 seconds
        
        def monitor_loop():
            logger.info("Memory monitor loop started")
            loop_count = 0
            while self.monitoring:
                try:
                    loop_count += 1
                    memory_info = self.get_memory_info()
                    
                    if not memory_info or 'process_memory_mb' not in memory_info:
                        logger.warning(f"get_memory_info returned invalid result: {memory_info}")
                        time.sleep(interval)
                        continue
                    
                    # Store in history
                    self.memory_history.append(memory_info['process_memory_mb'])
                    self.timestamp_history.append(memory_info['timestamp'])
                    
                    # Call callback if provided
                    if self.update_callback:
                        try:
                            self.update_callback(memory_info)
                            if loop_count == 1:
                                logger.info(f"First memory update sent: {memory_info['process_memory_mb']:.1f} MB")
                        except Exception as e:
                            logger.error(f"Error in memory monitor callback: {e}", exc_info=True)
                    else:
                        if loop_count == 1:
                            logger.warning("No callback provided for memory monitor")
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in memory monitor loop: {e}", exc_info=True)
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="MemoryMonitor")
        self.monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def get_history(self) -> tuple:
        """
        Get memory history data
        
        Returns:
            Tuple of (timestamps, memory_values)
        """
        return list(self.timestamp_history), list(self.memory_history)
    
    def force_gc(self):
        """Force garbage collection"""
        try:
            collected = gc.collect()
            logger.info(f"Forced garbage collection: {collected} objects collected")
            return collected
        except Exception as e:
            logger.error(f"Error forcing garbage collection: {e}")
            return 0
    
    def get_memory_summary(self) -> str:
        """
        Get a formatted memory summary string with detailed breakdown
        
        Returns:
            Formatted string with memory information
        """
        info = self.get_memory_info()
        breakdown = info.get('breakdown', {})
        
        summary = f"""Memory Usage Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Process Memory: {info['process_memory_mb']:.1f} MB ({info['process_memory_gb']:.2f} GB)
System Memory: {info['system_used_mb']:.1f} MB / {info['system_total_mb']:.1f} MB ({info['system_percent']:.1f}%)
System Available: {info['system_available_mb']:.1f} MB

Major Components:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EasyOCR Reader: {breakdown.get('easyocr_mb', 0):.1f} MB ({'Loaded' if breakdown.get('component_details', {}).get('easyocr_loaded') else 'Not Loaded'})
NumPy Arrays: {breakdown.get('numpy_arrays_mb', 0):.1f} MB ({breakdown.get('numpy_array_count', 0)} arrays)
PIL Images: {breakdown.get('pil_images_mb', 0):.1f} MB ({breakdown.get('pil_image_count', 0)} images)
GC Collections: {breakdown.get('gc_collections', 0)}
Total Objects: {breakdown.get('total_objects', 0)}
"""
        
        # Add large arrays info
        component_details = breakdown.get('component_details', {})
        if 'large_numpy_arrays' in component_details:
            large_arrays = component_details['large_numpy_arrays']
            summary += f"\n⚠️ Large Arrays (>1MB): {len(large_arrays)} found\n"
            for arr_info in large_arrays[:5]:  # Show top 5
                summary += f"  • {arr_info['size_mb']:.1f} MB - Shape: {arr_info['shape']}, Type: {arr_info['dtype']}\n"
        
        # Add top object types by size
        top_by_size = breakdown.get('top_object_types_by_size', {})
        if top_by_size:
            summary += "\nTop Memory Consumers (by size):\n"
            for obj_type, size_mb in list(top_by_size.items())[:5]:
                summary += f"  • {obj_type}: {size_mb:.2f} MB\n"
        
        # Add tracemalloc info
        tracemalloc_info = breakdown.get('tracemalloc', {})
        if tracemalloc_info and tracemalloc_info.get('top_consumers'):
            summary += "\nTop Memory Allocations:\n"
            for consumer in tracemalloc_info['top_consumers'][:5]:
                filename = consumer['file'].split('/')[-1] if '/' in consumer['file'] else consumer['file'].split('\\')[-1]
                summary += f"  • {filename}:{consumer['line']} - {consumer['size_mb']:.2f} MB ({consumer['count']} allocations)\n"
        
        # Add potential leak warning
        if 'potential_leak' in breakdown:
            summary += f"\n⚠️ WARNING: {breakdown['potential_leak']}\n"
        
        return summary
    
    def take_snapshot(self, label: str = "") -> Dict:
        """
        Take a memory snapshot for comparison
        
        Args:
            label: Label for this snapshot
            
        Returns:
            Snapshot dictionary
        """
        info = self.get_memory_info()
        snapshot = {
            'label': label or f"Snapshot_{len(self.memory_snapshots) + 1}",
            'timestamp': time.time(),
            'process_memory_mb': info['process_memory_mb'],
            'breakdown': info['breakdown']
        }
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def compare_snapshots(self, snapshot1: Dict, snapshot2: Dict) -> Dict:
        """
        Compare two memory snapshots to find what changed
        
        Args:
            snapshot1: First snapshot
            snapshot2: Second snapshot
            
        Returns:
            Dictionary with differences
        """
        diff = {
            'memory_change_mb': snapshot2['process_memory_mb'] - snapshot1['process_memory_mb'],
            'memory_change_percent': ((snapshot2['process_memory_mb'] - snapshot1['process_memory_mb']) / snapshot1['process_memory_mb'] * 100) if snapshot1['process_memory_mb'] > 0 else 0,
            'time_elapsed': snapshot2['timestamp'] - snapshot1['timestamp']
        }
        
        # Compare breakdowns
        breakdown1 = snapshot1.get('breakdown', {})
        breakdown2 = snapshot2.get('breakdown', {})
        
        diff['breakdown_changes'] = {
            'numpy_arrays_change': breakdown2.get('numpy_arrays_mb', 0) - breakdown1.get('numpy_arrays_mb', 0),
            'numpy_arrays_count_change': breakdown2.get('numpy_array_count', 0) - breakdown1.get('numpy_array_count', 0),
            'pil_images_change': breakdown2.get('pil_images_mb', 0) - breakdown1.get('pil_images_mb', 0),
        }
        
        return diff


# Global memory monitor instance
memory_monitor = MemoryMonitor()

