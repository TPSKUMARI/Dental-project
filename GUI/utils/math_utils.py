"""
Math utility functions for 3D operations
"""

import numpy as np
import math


def calculate_normal(v1, v2, v3):
    """
    Calculate normal vector for a triangle face
    
    Args:
        v1, v2, v3: The three vertices of the triangle (each a 3D point)
        
    Returns:
        Normalized normal vector
    """
    # Calculate edges
    edge1 = np.array(v2) - np.array(v1)
    edge2 = np.array(v3) - np.array(v1)
    
    # Calculate normal using cross product
    normal = np.cross(edge1, edge2)
    
    # Normalize the normal vector
    length = np.linalg.norm(normal)
    if length > 0:
        normal = normal / length
        
    return normal


def calculate_face_normals(vertices, faces):
    """
    Calculate normals for each face in the mesh
    
    Args:
        vertices: List of vertex coordinates
        faces: List of face indices
        
    Returns:
        List of face normals
    """
    face_normals = []
    
    for face in faces:
        if len(face) >= 3:
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            normal = calculate_normal(v1, v2, v3)
            face_normals.append(normal)
    
    return face_normals


def calculate_vertex_normals(vertices, faces):
    """
    Calculate averaged normals for each vertex based on connected faces
    
    Args:
        vertices: List of vertex coordinates
        faces: List of face indices
        
    Returns:
        List of vertex normals
    """
    # Initialize vertex normals with zeros
    vertex_normals = np.zeros((len(vertices), 3), dtype=np.float32)
    
    # For each face, calculate its normal and add to connected vertices
    for face in faces:
        if len(face) >= 3:
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            normal = calculate_normal(v1, v2, v3)
            
            # Add face normal to all vertices of this face
            for idx in face:
                vertex_normals[idx] += normal
    
    # Normalize all vertex normals
    for i in range(len(vertex_normals)):
        length = np.linalg.norm(vertex_normals[i])
        if length > 0:
            vertex_normals[i] = vertex_normals[i] / length
    
    return vertex_normals


def center_and_scale_model(vertices):
    """
    Center the model at origin and scale to fit in a 2x2x2 box
    
    Args:
        vertices: List of vertex coordinates
        
    Returns:
        Tuple containing:
        - Scaled vertices
        - Bounding box min coordinates
        - Bounding box max coordinates
    """
    # Convert to numpy array if it's not already
    vertices_array = np.array(vertices, dtype=np.float32)
    
    # Calculate bounding box
    bbox_min = np.min(vertices_array, axis=0)
    bbox_max = np.max(vertices_array, axis=0)
    
    # Calculate model dimensions
    dimensions = bbox_max - bbox_min
    max_dim = np.max(dimensions)
    
    # Scale to fit in a 2x2x2 box
    if max_dim > 0:
        scale_factor = 2.0 / max_dim
        scaled_vertices = vertices_array * scale_factor
        
        # Update bounding box
        bbox_min = np.min(scaled_vertices, axis=0)
        bbox_max = np.max(scaled_vertices, axis=0)
        
        return scaled_vertices, bbox_min, bbox_max
    
    return vertices_array, bbox_min, bbox_max
