"""
Model class for comparison results
"""

import numpy as np
from core.model import Model


class ComparisonModel(Model):
    """
    Class representing a comparison model with colored points
    showing the differences between two PLY models
    """

    def __init__(self):
        """Initialize an empty comparison model"""
        super(ComparisonModel, self).__init__()
        self.comparison_stats = {}
        self.changed_points_indices = None
        self.intensity_values = None

    def set_from_comparison(self, vertices, colors, intensity=None, changed_indices=None):
        """
        Set model data from comparison results

        Args:
            vertices: Array of vertex coordinates
            colors: Array of vertex colors
            intensity: Array of intensity values (optional)
            changed_indices: Indices of changed points (optional)
        """
        try:
            print(f"Setting comparison model from {len(vertices)} vertices")

            # Store comparison-specific data
            self.intensity_values = intensity
            self.changed_points_indices = changed_indices

            # Convert arrays to proper numpy arrays if they aren't already
            if vertices is not None and not isinstance(vertices, np.ndarray):
                vertices = np.array(vertices, dtype=np.float32)

            if colors is not None and not isinstance(colors, np.ndarray):
                colors = np.array(colors, dtype=np.float32)

            # Call parent method to set the basic model data
            self.set_data(vertices, None, colors, None, "Comparison Result")

            if changed_indices is not None:
                print(f"Number of changed points: {len(changed_indices)}")

        except Exception as e:
            print(f"Error in set_from_comparison: {e}")
            import traceback
            traceback.print_exc()

    def set_comparison_stats(self, stats):
        """Set comparison statistics"""
        try:
            print(f"Setting comparison stats: {stats}")
            self.comparison_stats = stats
        except Exception as e:
            print(f"Error setting comparison stats: {e}")
            import traceback
            traceback.print_exc()

    def get_comparison_stats(self):
        """Get comparison statistics"""
        return self.comparison_stats

    def get_changed_points(self):
        """
        Get vertices and colors of only the changed points

        Returns:
            Tuple containing (vertices, colors) of changed points
        """
        if self.changed_points_indices is None or len(self.changed_points_indices) == 0:
            return None, None

        changed_vertices = self.vertices[self.changed_points_indices]
        changed_colors = self.colors[self.changed_points_indices]

        return changed_vertices, changed_colors

    def get_intensity_range(self):
        """
        Get the range of intensity values

        Returns:
            Tuple containing (min, max) intensity
        """
        if self.intensity_values is None or len(self.intensity_values) == 0:
            return 0.0, 1.0

        return np.min(self.intensity_values), np.max(self.intensity_values)