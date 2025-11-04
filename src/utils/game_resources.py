"""
Game Resources Manager - Manages game UI templates and their metadata
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

from ..utils.logger import get_logger
from ..utils.template_matcher import TemplateMatcher

logger = get_logger(__name__)


class GameResource:
    """Represents a game UI resource (button, icon, etc.)"""
    
    def __init__(self, name: str, template_file: str, category: str = "general", 
                 description: str = "", clickable: bool = True):
        """
        Initialize game resource
        
        Args:
            name: Resource name (e.g., 'attack_button')
            template_file: Template image filename
            category: Resource category (e.g., 'button', 'icon', 'menu')
            description: Human-readable description
            clickable: Whether this resource can be clicked
        """
        self.name = name
        self.template_file = template_file
        self.category = category
        self.description = description
        self.clickable = clickable


class GameResourcesManager:
    """
    Manages game UI resources and provides easy access to find them
    """
    
    def __init__(self, templates_dir: str = "assets/templates"):
        """
        Initialize game resources manager
        
        Args:
            templates_dir: Directory containing template images
        """
        self.templates_dir = Path(templates_dir)
        self.template_matcher = TemplateMatcher(templates_dir)
        self.resources: Dict[str, GameResource] = {}
        
        # Load resources metadata if exists
        self._load_resources_metadata()
        
        logger.info(f"Game resources manager initialized with {len(self.resources)} resources")
    
    def register_resource(self, resource: GameResource):
        """
        Register a game resource
        
        Args:
            resource: GameResource instance
        """
        self.resources[resource.name] = resource
        logger.debug(f"Registered resource: {resource.name}")
    
    def find_resource(self, 
                     screenshot: np.ndarray,
                     resource_name: str,
                     confidence: float = None) -> Optional[Tuple[int, int, float]]:
        """
        Find a resource in a screenshot
        
        Args:
            screenshot: Screenshot to search in
            resource_name: Name of the resource to find
            confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (center_x, center_y, confidence) if found, None otherwise
        """
        if resource_name not in self.resources:
            logger.warning(f"Resource not registered: {resource_name}")
            return None
        
        resource = self.resources[resource_name]
        return self.template_matcher.find_template(
            screenshot, 
            resource.template_file,
            confidence
        )
    
    def find_resources_by_category(self, category: str) -> List[GameResource]:
        """
        Get all resources in a category
        
        Args:
            category: Category name
            
        Returns:
            List of GameResource objects
        """
        return [r for r in self.resources.values() if r.category == category]
    
    def _load_resources_metadata(self):
        """Load resources metadata from JSON file if exists"""
        metadata_file = self.templates_dir / "resources.json"
        
        if not metadata_file.exists():
            logger.debug("No resources metadata file found, using defaults")
            return
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data.get('resources', []):
                resource = GameResource(
                    name=item['name'],
                    template_file=item['template_file'],
                    category=item.get('category', 'general'),
                    description=item.get('description', ''),
                    clickable=item.get('clickable', True)
                )
                self.register_resource(resource)
                
            logger.info(f"Loaded {len(data.get('resources', []))} resources from metadata")
            
        except Exception as e:
            logger.error(f"Error loading resources metadata: {e}")
    
    def save_resources_metadata(self):
        """Save resources metadata to JSON file"""
        metadata_file = self.templates_dir / "resources.json"
        
        try:
            data = {
                'resources': [
                    {
                        'name': r.name,
                        'template_file': r.template_file,
                        'category': r.category,
                        'description': r.description,
                        'clickable': r.clickable
                    }
                    for r in self.resources.values()
                ]
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved resources metadata to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving resources metadata: {e}")

