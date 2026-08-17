"""
3D model data structures for PLY Viewer
"""

import numpy as np
from utils.math_utils import center_and_scale_model


class Model:
    """
    Class representing a 3D model loaded from a PLY file
    """
    
    def __init__(self):
        """Initialize an empty model"""
        self.vertices = None
        self.faces = None
        self.colors = None
        self.normals = None
        self.has_color = False
        self.bbox_min = None
        self.bbox_max = None
        self.filename = None
        
        # VBO objects for GPU acceleration
        self.vertex_vbo = None
        self.color_vbo = None
        self.normal_vbo = None
        self.index_vbo = None
        
        # Model statistics
        self.vertex_count = 0
        self.face_count = 0
        
    def is_loaded(self):
        """Check if the model has been loaded"""
        return self.vertices is not None and len(self.vertices) > 0
        
    def clear(self):
        """Clear all model data"""
        self.vertices = None
        self.faces = None
        self.colors = None
        self.normals = None
        self.has_color = False
        self.bbox_min = None
        self.bbox_max = None
        self.filename = None
        
        # Clear VBOs
        self.vertex_vbo = None
        self.color_vbo = None
        self.normal_vbo = None
        self.index_vbo = None
        
        self.vertex_count = 0
        self.face_count = 0
        
    def set_data(self, vertices, faces=None, colors=None, normals=None, filename=None):
        """
        Set model data
        
        Args:
            vertices: List of vertex coordinates
            faces: List of face indices (optional)
            colors: List of vertex colors (optional)
            normals: List of vertex normals (optional)
            filename: Source filename (optional)
        """
        self.clear()
        
        # Store basic data
        self.vertices = np.array(vertices, dtype=np.float32)
        self.vertex_count = len(self.vertices)
        
        if faces is not None:
            self.faces = faces
            self.face_count = len(self.faces)
        
        if colors is not None:
            self.colors = np.array(colors, dtype=np.float32)
            self.has_color = True
        
        if normals is not None:
            self.normals = np.array(normals, dtype=np.float32)
            
        self.filename = filename
        
        # Center and scale the model
        self.center_and_scale()
        
    def center_and_scale(self):
        """Center and scale the model to fit in view"""
        if self.vertices is not None and len(self.vertices) > 0:
            self.vertices, self.bbox_min, self.bbox_max = center_and_scale_model(self.vertices)
            
    def get_stats(self):
        """Get model statistics as a dictionary"""
        return {
            "filename": self.filename,
            "vertices": self.vertex_count,
            "faces": self.face_count,
            "has_colors": self.has_color,
            "has_normals": self.normals is not None,
            "dimensions": self.bbox_max - self.bbox_min if self.bbox_max is not None else None
        }
