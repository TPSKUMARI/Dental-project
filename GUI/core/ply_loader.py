"""
PLY file loading utilities for PLY Viewer
"""

import numpy as np
from plyfile import PlyData
from utils.math_utils import calculate_vertex_normals
from core.model import Model


def load_ply_file(filename):
    """
    Load a PLY file and create a Model instance
    
    Args:
        filename: Path to the PLY file
        
    Returns:
        Model instance or None if loading failed
    """
    try:
        # Create a new model
        model = Model()
        
        # Read PLY file
        plydata = PlyData.read(filename)
        
        # Extract vertices
        vertex_data = plydata['vertex']
        vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        
        # Check for colors
        colors = None
        has_color = False
        if 'red' in vertex_data and 'green' in vertex_data and 'blue' in vertex_data:
            colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
            has_color = True
            
        # Check for normals
        normals = None
        if 'nx' in vertex_data and 'ny' in vertex_data and 'nz' in vertex_data:
            normals = np.vstack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']]).T
            
        # Extract faces if available
        faces = []
        if 'face' in plydata:
            face_data = plydata['face']
            if 'vertex_indices' in face_data:
                faces = [face for face in face_data['vertex_indices']]
            elif 'vertex_index' in face_data:  # Some PLY files use this format
                faces = [face for face in face_data['vertex_index']]
                
        # Calculate normals if not provided
        if normals is None and len(faces) > 0:
            normals = calculate_vertex_normals(vertices, faces)
            
        # Set the model data
        model.set_data(vertices, faces, colors, normals, filename)
        
        return model
        
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None
