"""
Enhanced Integrated Dental Segmentation Application - Combines teeth/gum segmentation
with surface separation and ENHANCED multi-region marking capabilities adapted from HTML version.
"""

import os
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox,
    QCheckBox, QSlider, QFormLayout, QProgressBar, QSplitter,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QListWidget,
    QListWidgetItem, QInputDialog, QScrollArea, QLineEdit, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

from core.gl_widget import GLWidget
from core.ply_loader import load_ply_file
from core.model import Model
from core.comparison_model import ComparisonModel
from utils.gl_utils import configure_gl_format
from ui.style import apply_application_style

from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


class EnhancedInteractiveGLWidget(GLWidget):
    """Enhanced GL widget with multi-region interactive marking capabilities"""

    # Signal emitted when user clicks on a point
    point_clicked = pyqtSignal(int)  # vertex index

    def __init__(self):
        super().__init__()
        self.marking_enabled = False
        self.brush_radius = 5.0
        self.marking_surface = None  # 'inner' or 'outer'

        # ENHANCED: Multi-region support
        self.current_selection = set()  # Currently being marked
        self.saved_regions = {}  # {region_name: {'indices': set, 'color': [r,g,b], 'surface_type': str, 'visible': bool}}
        self.region_colors = [
            [255, 107, 107],  # Red
            [78, 205, 196],  # Teal
            [69, 183, 209],  # Blue
            [249, 202, 36],  # Yellow
            [108, 92, 231],  # Purple
            [165, 94, 234],  # Pink
            [38, 222, 129],  # Green
            [253, 121, 168],  # Pink
            [225, 112, 85],  # Orange
            [0, 184, 148]  # Emerald
        ]
        self.region_counter = 0

        self.surface_indices = None
        self.is_dragging = False
        self.last_mouse_pos = None
        self.base_model = None
        self.marking_model = None

    def enable_marking(self, surface_type, surface_indices, brush_radius=5.0):
        """Enable marking mode for a specific surface"""
        self.marking_enabled = True
        self.marking_surface = surface_type
        self.surface_indices = set(surface_indices) if surface_indices is not None else set()
        self.brush_radius = brush_radius
        self.setCursor(Qt.CrossCursor)

        if self.model:
            self.base_model = self.model

        self._update_marking_visualization()

    def disable_marking(self):
        """Disable marking mode but preserve all regions"""
        self.marking_enabled = False
        self.marking_surface = None
        self.surface_indices = None
        self.is_dragging = False
        self.setCursor(Qt.ArrowCursor)

        # Keep all saved regions visible
        self._update_marking_visualization()

    def save_current_selection_as_region(self, region_name, description=""):
        """Save current selection as a named region"""
        if len(self.current_selection) == 0:
            return False

        # Check if region name already exists
        if region_name in self.saved_regions:
            return False

        # Assign color
        color = self.region_colors[self.region_counter % len(self.region_colors)]
        self.region_counter += 1

        # Save region
        self.saved_regions[region_name] = {
            'indices': self.current_selection.copy(),
            'color': color,
            'surface_type': self.marking_surface,
            'description': description,
            'point_count': len(self.current_selection),
            'visible': True
        }

        # Clear current selection
        self.current_selection.clear()
        self._update_marking_visualization()
        return True

    def delete_region(self, region_name):
        """Delete a saved region"""
        if region_name in self.saved_regions:
            del self.saved_regions[region_name]
            self._update_marking_visualization()
            return True
        return False

    def toggle_region_visibility(self, region_name):
        """Toggle visibility of a region"""
        if region_name in self.saved_regions:
            self.saved_regions[region_name]['visible'] = not self.saved_regions[region_name]['visible']
            self._update_marking_visualization()

    def show_all_regions(self, visible=True):
        """Show or hide all regions"""
        for region_data in self.saved_regions.values():
            region_data['visible'] = visible
        self._update_marking_visualization()

    def get_region_data(self, region_name):
        """Get data for a specific region"""
        return self.saved_regions.get(region_name, None)

    def get_all_regions(self):
        """Get all saved regions"""
        return self.saved_regions

    def clear_current_selection(self):
        """Clear current selection"""
        self.current_selection.clear()
        self._update_marking_visualization()

    def clear_all_regions(self):
        """Clear all saved regions"""
        self.saved_regions.clear()
        self.current_selection.clear()
        self.region_counter = 0
        self._update_marking_visualization()

    def set_brush_size(self, size):
        """Update brush size"""
        self.brush_radius = size

    def mousePressEvent(self, event):
        """Handle mouse press events for marking"""
        if self.marking_enabled and event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = (event.x(), event.y())
            self._mark_at_position(event.x(), event.y())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for continuous marking"""
        if self.marking_enabled and self.is_dragging and (event.buttons() & Qt.LeftButton):
            if not hasattr(self, 'last_mark_time'):
                self.last_mark_time = 0

            import time
            current_time = time.time()

            if current_time - self.last_mark_time > 0.05:  # 50ms throttle
                self._mark_at_position(event.x(), event.y())
                self.last_mouse_pos = (event.x(), event.y())
                self.last_mark_time = current_time
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.marking_enabled and event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.last_mouse_pos = None
        else:
            super().mouseReleaseEvent(event)

    def _mark_at_position(self, x, y):
        """Mark points at the given screen position"""
        if not self.base_model or not self.surface_indices:
            return

        picked_indices = self._handle_selection_like_web_viewer(x, y)
        if picked_indices:
            old_count = len(self.current_selection)
            self.current_selection.update(picked_indices)
            new_count = len(self.current_selection)

            print(f"Marked {new_count - old_count} new points at position ({x}, {y})")
            print(f"Total current selection: {new_count} points on {self.marking_surface} surface")

            self._update_marking_visualization()

    def _handle_selection_like_web_viewer(self, screen_x, screen_y):
        """Selection method with accurate coordinate transformation"""
        widget_width = self.width()
        widget_height = self.height()

        if widget_width == 0 or widget_height == 0:
            return []

        mouse_x = (screen_x / widget_width) * 2.0 - 1.0
        mouse_y = 1.0 - (screen_y / widget_height) * 2.0

        intersect_point = self._accurate_ray_intersect(mouse_x, mouse_y)

        if intersect_point is not None:
            selected_indices = self._select_points_near_web_method(intersect_point, self.brush_radius)
            return selected_indices

        return []

    def _accurate_ray_intersect(self, mouse_x, mouse_y):
        """Accurate ray intersection using current camera parameters"""
        try:
            parent_widget = self
            rotation_x = getattr(parent_widget, 'rotation_x', 0.0)
            rotation_y = getattr(parent_widget, 'rotation_y', 0.0)
            zoom = getattr(parent_widget, 'zoom', -3.0)

            vertices = self.base_model.vertices
            surface_indices_list = list(self.surface_indices)
            surface_vertices = vertices[surface_indices_list]

            if len(surface_vertices) == 0:
                return None

            model_center = np.mean(vertices, axis=0)
            model_bounds = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            model_size = np.max(model_bounds)

            camera_distance = abs(zoom) * model_size * 0.5
            base_camera_pos = np.array([0, 0, camera_distance])

            import math
            rx_rad = math.radians(rotation_x)
            ry_rad = math.radians(rotation_y)

            cos_x, sin_x = math.cos(rx_rad), math.sin(rx_rad)
            cos_y, sin_y = math.cos(ry_rad), math.sin(ry_rad)

            cam_x = base_camera_pos[0] * cos_y - base_camera_pos[2] * sin_y
            cam_z = base_camera_pos[0] * sin_y + base_camera_pos[2] * cos_y
            base_camera_pos[0] = cam_x
            base_camera_pos[2] = cam_z

            cam_y = base_camera_pos[1] * cos_x - base_camera_pos[2] * sin_x
            cam_z = base_camera_pos[1] * sin_x + base_camera_pos[2] * cos_x
            base_camera_pos[1] = cam_y
            base_camera_pos[2] = cam_z

            camera_pos = base_camera_pos + model_center

            aspect_ratio = self.width() / self.height() if self.height() > 0 else 1.0
            fov_rad = math.radians(45.0)

            tan_half_fov = math.tan(fov_rad / 2.0)
            ray_x = mouse_x * tan_half_fov * aspect_ratio
            ray_y = mouse_y * tan_half_fov
            ray_z = -1.0

            ray_dir = np.array([ray_x, ray_y, ray_z])
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            temp_x = ray_dir[0] * cos_y - ray_dir[2] * sin_y
            temp_z = ray_dir[0] * sin_y + ray_dir[2] * cos_y
            ray_dir[0] = temp_x
            ray_dir[2] = temp_z

            temp_y = ray_dir[1] * cos_x - ray_dir[2] * sin_x
            temp_z = ray_dir[1] * sin_x + ray_dir[2] * cos_x
            ray_dir[1] = temp_y
            ray_dir[2] = temp_z

            min_distance = float('inf')
            closest_point = None
            tolerance = model_size * 0.05

            for i, surface_idx in enumerate(surface_indices_list):
                point = vertices[surface_idx]
                to_point = point - camera_pos
                projection_length = np.dot(to_point, ray_dir)

                if projection_length <= 0:
                    continue

                closest_on_ray = camera_pos + projection_length * ray_dir
                distance_to_ray = np.linalg.norm(point - closest_on_ray)

                if distance_to_ray < min_distance:
                    min_distance = distance_to_ray
                    closest_point = point

            if min_distance < tolerance and closest_point is not None:
                return closest_point
            else:
                return self._fallback_screen_intersection(mouse_x, mouse_y)

        except Exception as e:
            print(f"Ray intersection error: {e}")
            return self._fallback_screen_intersection(mouse_x, mouse_y)

    def _fallback_screen_intersection(self, mouse_x, mouse_y):
        """Fallback method using simple screen space projection"""
        try:
            vertices = self.base_model.vertices
            surface_indices_list = list(self.surface_indices)
            surface_vertices = vertices[surface_indices_list]

            if len(surface_vertices) == 0:
                return None

            model_center = np.mean(vertices, axis=0)
            projected_points = []

            for i, surface_idx in enumerate(surface_indices_list):
                point = vertices[surface_idx]
                centered = point - model_center
                model_size = np.max(np.ptp(vertices, axis=0))
                proj_x = centered[0] / (model_size * 0.5)
                proj_y = centered[1] / (model_size * 0.5)

                distance = np.sqrt((proj_x - mouse_x) ** 2 + (proj_y - mouse_y) ** 2)
                projected_points.append((distance, point, surface_idx))

            if projected_points:
                projected_points.sort(key=lambda x: x[0])
                closest_distance, closest_point, closest_idx = projected_points[0]

                if closest_distance < 0.2:
                    return closest_point

            return None

        except Exception as e:
            print(f"Fallback intersection error: {e}")
            return None

    def _select_points_near_web_method(self, center_point, radius):
        """Point selection method with better spatial accuracy"""
        selected_indices = []
        vertices = self.base_model.vertices
        surface_indices_list = list(self.surface_indices)

        surface_vertices = vertices[surface_indices_list]

        try:
            kdtree = KDTree(surface_vertices)
            indices_within_radius = kdtree.query_ball_point(center_point, radius)

            for local_idx in indices_within_radius:
                if local_idx < len(surface_indices_list):
                    original_idx = surface_indices_list[local_idx]
                    selected_indices.append(original_idx)

        except ImportError:
            for i, surface_idx in enumerate(surface_indices_list):
                point = vertices[surface_idx]
                distance = np.sqrt(
                    np.power(point[0] - center_point[0], 2) +
                    np.power(point[1] - center_point[1], 2) +
                    np.power(point[2] - center_point[2], 2)
                )
                if distance <= radius:
                    selected_indices.append(surface_idx)

        if len(selected_indices) > 5000:
            selected_indices = selected_indices[:5000]

        return selected_indices

    def _update_marking_visualization(self):
        """Update visualization with current selection and all saved regions"""
        if not self.base_model:
            return

        try:
            vertices = self.base_model.vertices

            if hasattr(self.base_model, 'colors') and self.base_model.colors is not None:
                colors = np.copy(self.base_model.colors) / 255.0
            else:
                colors = np.ones((len(vertices), 3)) * 0.7

            # Apply original colors as base
            for i in range(len(vertices)):
                if hasattr(self.base_model, 'colors') and self.base_model.colors is not None:
                    colors[i] = self.base_model.colors[i] / 255.0
                else:
                    colors[i] = [0.3, 0.6, 0.9]  # Default blue

            # Apply saved regions (if visible)
            for region_name, region_data in self.saved_regions.items():
                if region_data['visible']:
                    region_color = np.array(region_data['color']) / 255.0
                    for idx in region_data['indices']:
                        colors[idx] = region_color

            # Apply current selection (bright yellow - highest priority)
            for idx in self.current_selection:
                colors[idx] = [1.0, 1.0, 0.2]  # Bright yellow

            # Update the model
            if not hasattr(self, 'marking_model') or self.marking_model is None:
                self.marking_model = ComparisonModel()

            self.marking_model.set_data(vertices, None, colors * 255, None, "Enhanced Marking Visualization")
            self.set_model(self.marking_model)
            self.update()

        except Exception as e:
            print(f"Marking visualization error: {e}")

    def get_current_selection_count(self):
        """Get count of currently selected points"""
        return len(self.current_selection)

    def get_region_count(self):
        """Get count of saved regions"""
        return len(self.saved_regions)

    def get_selection_percentage(self):
        """Get percentage of surface that is currently selected"""
        if not self.surface_indices or len(self.surface_indices) == 0:
            return 0.0
        return (len(self.current_selection) / len(self.surface_indices)) * 100.0


class EnhancedSurfaceMarker:
    """Enhanced surface marker with multi-region capabilities"""

    def __init__(self):
        self.original_vertices = None

    def store_original_vertices(self, original_vertices):
        """Store the original vertices before any normalization"""
        self.original_vertices = original_vertices.copy()

    def extract_region_point_cloud(self, vertices, colors, region_indices, region_name):
        """Extract region as point cloud with original coordinates"""
        if len(region_indices) == 0:
            return None

        try:
            original_vertices = self.original_vertices if self.original_vertices is not None else vertices
            region_vertices = original_vertices[list(region_indices)]
            region_colors = colors[list(region_indices)] if colors is not None else None

            bounds = {
                'min_x': float(np.min(region_vertices[:, 0])),
                'max_x': float(np.max(region_vertices[:, 0])),
                'min_y': float(np.min(region_vertices[:, 1])),
                'max_y': float(np.max(region_vertices[:, 1])),
                'min_z': float(np.min(region_vertices[:, 2])),
                'max_z': float(np.max(region_vertices[:, 2]))
            }

            center = {
                'x': float(np.mean(region_vertices[:, 0])),
                'y': float(np.mean(region_vertices[:, 1])),
                'z': float(np.mean(region_vertices[:, 2]))
            }

            point_cloud = {
                'name': region_name,
                'point_count': len(region_indices),
                'vertices': region_vertices.copy(),
                'colors': region_colors.copy() if region_colors is not None else None,
                'indices': list(region_indices),
                'bounds': bounds,
                'center': center,
                'coordinate_type': 'original' if self.original_vertices is not None else 'normalized'
            }

            return point_cloud

        except Exception as e:
            print(f"Error extracting point cloud: {e}")
            return None

    def save_point_cloud_as_ply(self, point_cloud, filename):
        """Save point cloud to PLY file"""
        if not point_cloud or 'vertices' not in point_cloud:
            return False

        try:
            vertices = point_cloud['vertices']
            colors = point_cloud['colors']
            point_count = len(vertices)

            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"comment Point cloud: {point_cloud['name']}\n")
                f.write(f"element vertex {point_count}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")

                if colors is not None:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")

                f.write("end_header\n")

                for i in range(point_count):
                    x, y, z = vertices[i]

                    if colors is not None:
                        if np.max(colors[i]) <= 1.0:
                            r, g, b = (colors[i] * 255).astype(int)
                        else:
                            r, g, b = colors[i].astype(int)
                        f.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")
                    else:
                        f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")

            return True

        except Exception as e:
            print(f"Error saving PLY file: {e}")
            return False

    def save_regions_combined_csv(self, all_regions_data, filename):
        """Save all regions to a combined CSV file with region labels"""
        try:
            with open(filename, 'w') as f:
                f.write("X,Y,Z,R,G,B,Region,RegionColor\n")

                for region_name, region_data in all_regions_data.items():
                    point_cloud = region_data['point_cloud']
                    region_color_hex = '#{:02x}{:02x}{:02x}'.format(*region_data['color'])

                    vertices = point_cloud['vertices']
                    colors = point_cloud['colors']

                    for i in range(len(vertices)):
                        x, y, z = vertices[i]

                        if colors is not None:
                            if np.max(colors[i]) <= 1.0:
                                r, g, b = (colors[i] * 255).astype(int)
                            else:
                                r, g, b = colors[i].astype(int)
                        else:
                            r = g = b = 128  # Default gray

                        f.write(f"{x:.8f},{y:.8f},{z:.8f},{r},{g},{b},{region_name},{region_color_hex}\n")

            return True
        except Exception as e:
            print(f"Error saving combined CSV: {e}")
            return False


class EnhancedIntegratedDentalSegmentationWindow(QMainWindow):
    """Enhanced main window with multi-region marking capabilities"""

    def __init__(self):
        super().__init__()
        apply_application_style()

        self.model = None
        self.segmenter = IntegratedDentalSegmenter()  # Keep existing segmenter
        self.enhanced_surface_marker = EnhancedSurfaceMarker()

        self.teeth_gum_segmented = None
        self.surface_separated = None

        self.current_brush_radius = 0.15
        self.marking_enabled = False
        self.selected_region_name = None

        self.setup_ui()
        self.setWindowTitle("Enhanced Dental Segmentation - Multi-Region Marking")
        self.resize(1600, 1000)

    def setup_ui(self):
        """Set up the enhanced user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("Enhanced Dental Segmentation - Multi-Region Marking")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Main content splitter
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Create visualization area
        self.create_visualization_area()

        # Enhanced control panel
        self.control_scroll_area = self.create_enhanced_control_panel()
        self.main_splitter.addWidget(self.control_scroll_area)

        # Set splitter sizes
        self.main_splitter.setSizes([1000, 600])

        main_layout.addWidget(self.main_splitter)

        # Status bar
        self.statusBar().showMessage("Ready - Load a dental model to begin")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def create_visualization_area(self):
        """Create the visualization area"""
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        # Original model view
        original_group = QGroupBox("Original Model")
        original_layout = QVBoxLayout(original_group)
        self.original_gl_widget = GLWidget()
        self.original_gl_widget.setMinimumSize(400, 200)
        original_layout.addWidget(self.original_gl_widget)

        # Teeth/Gum segmentation view
        teeth_gum_group = QGroupBox("STEP 1: Teeth/Gum Segmentation")
        teeth_gum_layout = QVBoxLayout(teeth_gum_group)
        self.teeth_gum_gl_widget = GLWidget()
        self.teeth_gum_gl_widget.setMinimumSize(400, 200)
        teeth_gum_layout.addWidget(self.teeth_gum_gl_widget)

        # Enhanced surface separation and multi-region marking view
        surface_group = QGroupBox("STEP 2: Surface Separation + STEP 3: Multi-Region Marking")
        surface_layout = QVBoxLayout(surface_group)
        self.surface_gl_widget = EnhancedInteractiveGLWidget()  # Use enhanced widget
        self.surface_gl_widget.setMinimumSize(400, 350)
        surface_layout.addWidget(self.surface_gl_widget)

        viz_layout.addWidget(original_group)
        viz_layout.addWidget(teeth_gum_group)
        viz_layout.addWidget(surface_group)

        self.main_splitter.addWidget(viz_widget)

    def create_enhanced_control_panel(self):
        """Create the enhanced control panel with multi-region support"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(500)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(15, 15, 15, 15)

        # Load Model Section (keep existing)
        load_group = QGroupBox("Load Model")
        load_layout = QVBoxLayout(load_group)

        self.load_btn = QPushButton("Load Dental Model")
        self.load_btn.clicked.connect(self.load_model)
        load_layout.addWidget(self.load_btn)

        self.model_info_label = QLabel("No model loaded")
        self.model_info_label.setWordWrap(True)
        load_layout.addWidget(self.model_info_label)

        control_layout.addWidget(load_group)

        # Step 1: Teeth/Gum Segmentation (keep existing)
        step1_group = QGroupBox("Step 1: Teeth/Gum Segmentation")
        step1_layout = QVBoxLayout(step1_group)

        self.auto_refine_cb = QCheckBox("Auto-refine segmentation")
        self.auto_refine_cb.setChecked(True)
        step1_layout.addWidget(self.auto_refine_cb)

        self.segment_teeth_gum_btn = QPushButton("Run Teeth/Gum Segmentation")
        self.segment_teeth_gum_btn.clicked.connect(self.run_teeth_gum_segmentation)
        self.segment_teeth_gum_btn.setEnabled(False)
        step1_layout.addWidget(self.segment_teeth_gum_btn)

        self.step1_result_label = QLabel("Step 1 not completed")
        self.step1_result_label.setWordWrap(True)
        step1_layout.addWidget(self.step1_result_label)

        control_layout.addWidget(step1_group)

        # Step 2: Surface Separation (keep existing)
        step2_group = QGroupBox("Step 2: Surface Separation")
        step2_layout = QFormLayout(step2_group)

        normal_layout = QHBoxLayout()
        self.normal_slider = QSlider(Qt.Horizontal)
        self.normal_slider.setRange(1, 9)
        self.normal_slider.setValue(4)
        self.normal_slider.valueChanged.connect(self.update_normal_threshold)
        normal_layout.addWidget(self.normal_slider)
        self.normal_label = QLabel("0.4")
        normal_layout.addWidget(self.normal_label)
        step2_layout.addRow("Normal Sensitivity:", normal_layout)

        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(100, 5000)
        self.min_size_spinbox.setValue(500)
        step2_layout.addRow("Min Surface Size:", self.min_size_spinbox)

        self.separate_surfaces_btn = QPushButton("Run Surface Separation")
        self.separate_surfaces_btn.clicked.connect(self.run_surface_separation)
        self.separate_surfaces_btn.setEnabled(False)

        step2_widget = QWidget()
        step2_widget_layout = QVBoxLayout(step2_widget)
        step2_widget_layout.addWidget(step2_group)
        step2_widget_layout.addWidget(self.separate_surfaces_btn)

        self.step2_result_label = QLabel("Step 2 not completed")
        self.step2_result_label.setWordWrap(True)
        step2_widget_layout.addWidget(self.step2_result_label)

        control_layout.addWidget(step2_widget)

        # ENHANCED Step 3: Multi-Region Interactive Marking
        marking_group = QGroupBox("Step 3: Multi-Region Surface Marking")
        marking_layout = QVBoxLayout(marking_group)

        # Surface selection
        surface_select_layout = QHBoxLayout()
        surface_select_layout.addWidget(QLabel("Mark on:"))
        self.surface_combo = QComboBox()
        self.surface_combo.addItems(["Inner Teeth Surface", "Outer Teeth Surface"])
        surface_select_layout.addWidget(self.surface_combo)
        marking_layout.addLayout(surface_select_layout)

        # Brush size
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Brush Size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(5, 100)
        self.brush_slider.setValue(15)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        brush_layout.addWidget(self.brush_slider)
        self.brush_label = QLabel("0.15")
        brush_layout.addWidget(self.brush_label)
        marking_layout.addLayout(brush_layout)

        # Current selection controls
        selection_controls_layout = QHBoxLayout()
        self.enable_marking_btn = QPushButton("Start Marking")
        self.enable_marking_btn.clicked.connect(self.toggle_marking)
        self.enable_marking_btn.setEnabled(False)
        selection_controls_layout.addWidget(self.enable_marking_btn)

        self.clear_current_btn = QPushButton("Clear Current")
        self.clear_current_btn.clicked.connect(self.clear_current_selection)
        self.clear_current_btn.setEnabled(False)
        selection_controls_layout.addWidget(self.clear_current_btn)
        marking_layout.addLayout(selection_controls_layout)

        # Region name input
        region_name_layout = QHBoxLayout()
        region_name_layout.addWidget(QLabel("Region Name:"))
        self.region_name_input = QLineEdit()
        self.region_name_input.setPlaceholderText("e.g., Upper Molars, Lower Incisors")
        self.region_name_input.textChanged.connect(self.on_region_name_changed)
        region_name_layout.addWidget(self.region_name_input)
        marking_layout.addLayout(region_name_layout)

        # Save/Cancel region
        save_controls_layout = QHBoxLayout()
        self.save_region_btn = QPushButton("Save as Region")
        self.save_region_btn.clicked.connect(self.save_current_as_region)
        self.save_region_btn.setEnabled(False)
        self.save_region_btn.setStyleSheet("""
            QPushButton {
                background: linear-gradient(135deg, #27ae60, #2ecc71);
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: linear-gradient(135deg, #2ecc71, #27ae60);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        save_controls_layout.addWidget(self.save_region_btn)

        self.cancel_region_btn = QPushButton("Cancel")
        self.cancel_region_btn.clicked.connect(self.cancel_current_region)
        self.cancel_region_btn.setEnabled(False)
        save_controls_layout.addWidget(self.cancel_region_btn)
        marking_layout.addLayout(save_controls_layout)

        # Current selection info
        self.current_selection_info_label = QLabel("")
        self.current_selection_info_label.setWordWrap(True)
        self.current_selection_info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(155, 89, 182, 0.1);
                border: 1px solid rgba(155, 89, 182, 0.3);
                border-radius: 6px;
                padding: 8px;
                color: #8e44ad;
                font-size: 12px;
            }
        """)
        marking_layout.addWidget(self.current_selection_info_label)

        # Instructions
        instructions_label = QLabel("""
        <b>Multi-Region Marking:</b><br>
        • Select surface type: Inner (GREEN) or Outer (MAGENTA)<br>
        • Click "Start Marking" to enter selection mode<br>
        • Click and drag on colored areas to mark regions<br>
        • Enter descriptive region name<br>
        • Click "Save as Region" to preserve the marked area<br>
        • Repeat for multiple regions on the same surface<br>
        • Each region gets a unique color automatically
        """)
        instructions_label.setWordWrap(True)
        instructions_label.setStyleSheet("""
            QLabel {
                background-color: rgba(52, 152, 219, 0.1);
                border: 1px solid rgba(52, 152, 219, 0.3);
                border-radius: 6px;
                padding: 10px;
                color: #2980b9;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        marking_layout.addWidget(instructions_label)

        control_layout.addWidget(marking_group)

        # ENHANCED Step 4: Region Management
        regions_group = QGroupBox("Step 4: Saved Regions Management")
        regions_layout = QVBoxLayout(regions_group)

        # Regions list with enhanced display
        regions_list_label = QLabel("Saved Regions:")
        regions_list_label.setFont(QFont("Arial", 10, QFont.Bold))
        regions_layout.addWidget(regions_list_label)

        self.regions_list = QListWidget()
        self.regions_list.setMaximumHeight(200)
        self.regions_list.itemSelectionChanged.connect(self.on_region_selected)
        self.regions_list.itemDoubleClicked.connect(self.toggle_selected_region_visibility)
        regions_layout.addWidget(self.regions_list)

        # Region management controls
        region_mgmt_layout1 = QHBoxLayout()

        self.show_hide_region_btn = QPushButton("Show/Hide Region")
        self.show_hide_region_btn.clicked.connect(self.toggle_selected_region_visibility)
        self.show_hide_region_btn.setEnabled(False)
        region_mgmt_layout1.addWidget(self.show_hide_region_btn)

        self.delete_region_btn = QPushButton("Delete Region")
        self.delete_region_btn.clicked.connect(self.delete_selected_region)
        self.delete_region_btn.setEnabled(False)
        region_mgmt_layout1.addWidget(self.delete_region_btn)

        regions_layout.addLayout(region_mgmt_layout1)

        region_mgmt_layout2 = QHBoxLayout()

        self.show_all_regions_btn = QPushButton("Show All Regions")
        self.show_all_regions_btn.clicked.connect(self.show_all_regions)
        self.show_all_regions_btn.setEnabled(False)
        region_mgmt_layout2.addWidget(self.show_all_regions_btn)

        self.hide_all_regions_btn = QPushButton("Hide All Regions")
        self.hide_all_regions_btn.clicked.connect(self.hide_all_regions)
        self.hide_all_regions_btn.setEnabled(False)
        region_mgmt_layout2.addWidget(self.hide_all_regions_btn)

        regions_layout.addLayout(region_mgmt_layout2)

        # Export controls
        export_layout1 = QHBoxLayout()

        self.export_selected_region_btn = QPushButton("Export Selected Region")
        self.export_selected_region_btn.clicked.connect(self.export_selected_region)
        self.export_selected_region_btn.setEnabled(False)
        export_layout1.addWidget(self.export_selected_region_btn)

        self.export_all_regions_btn = QPushButton("Export All Regions")
        self.export_all_regions_btn.clicked.connect(self.export_all_regions)
        self.export_all_regions_btn.setEnabled(False)
        export_layout1.addWidget(self.export_all_regions_btn)

        regions_layout.addLayout(export_layout1)

        # Clear all regions
        self.clear_all_regions_btn = QPushButton("Clear All Regions")
        self.clear_all_regions_btn.clicked.connect(self.clear_all_regions)
        self.clear_all_regions_btn.setEnabled(False)
        self.clear_all_regions_btn.setStyleSheet("""
            QPushButton {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: linear-gradient(135deg, #c0392b, #e74c3c);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        regions_layout.addWidget(self.clear_all_regions_btn)

        control_layout.addWidget(regions_group)

        # Visualization Controls (keep existing)
        viz_group = QGroupBox("Visualization Controls")
        viz_layout = QVBoxLayout(viz_group)

        self.show_points_cb = QCheckBox("Show Points")
        self.show_points_cb.setChecked(True)
        self.show_points_cb.toggled.connect(self.toggle_points)
        viz_layout.addWidget(self.show_points_cb)

        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("Point Size:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(3)
        self.point_size_slider.valueChanged.connect(self.change_point_size)
        point_size_layout.addWidget(self.point_size_slider)
        self.point_size_label = QLabel("3")
        point_size_layout.addWidget(self.point_size_label)
        viz_layout.addLayout(point_size_layout)

        self.reset_view_btn = QPushButton("Reset All Views")
        self.reset_view_btn.clicked.connect(self.reset_view)
        viz_layout.addWidget(self.reset_view_btn)

        control_layout.addWidget(viz_group)

        # Enhanced Color Legend
        legend_group = QGroupBox("Enhanced Color Legend")
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.addWidget(QLabel("Step 1: Teeth/Gum Colors:"))
        legend_layout.addWidget(QLabel("⬜ White: Teeth  🟥 Red: Gums  ⬛ Gray: Other"))
        legend_layout.addWidget(QLabel(""))
        legend_layout.addWidget(QLabel("Step 2: Surface Colors:"))
        legend_layout.addWidget(QLabel("🟢 GREEN: Inner Teeth  🟣 MAGENTA: Outer Teeth"))
        legend_layout.addWidget(QLabel(""))
        legend_layout.addWidget(QLabel("Step 3: Region Marking:"))
        legend_layout.addWidget(QLabel("🟡 YELLOW: Current Selection"))
        legend_layout.addWidget(QLabel("🎨 Various Colors: Saved Regions"))
        legend_layout.addWidget(QLabel("💡 Double-click region to toggle visibility"))
        control_layout.addWidget(legend_group)

        control_layout.addStretch()

        # Set up timer for updating selection info
        self.selection_timer = QTimer()
        self.selection_timer.timeout.connect(self.update_current_selection_info)

        scroll_area.setWidget(control_widget)
        return scroll_area

    # Keep existing methods from original class...
    def load_model(self):
        """Load a dental model"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Dental PLY File", "", "PLY Files (*.ply)"
        )

        if not file_name:
            return

        self.statusBar().showMessage(f"Loading model: {file_name}...")
        self.progress_bar.setVisible(True)

        model = load_ply_file(file_name)

        if model:
            self.model = model
            self.segmenter.set_model(model)

            # Store original vertices in enhanced surface marker
            if model.vertices is not None:
                self.enhanced_surface_marker.store_original_vertices(model.vertices)

            self.original_gl_widget.set_model(model)
            self.original_gl_widget.show_points = True
            self.original_gl_widget.update()

            vertex_count = len(model.vertices) if model.vertices is not None else 0
            face_count = len(model.faces) if hasattr(model, 'faces') and model.faces is not None else 0
            has_color = "Yes" if model.has_color else "No"

            self.model_info_label.setText(
                f"Loaded: {os.path.basename(file_name)}\n"
                f"Vertices: {vertex_count:,}\n"
                f"Faces: {face_count:,}\n"
                f"Has colors: {has_color}"
            )
            self.statusBar().showMessage(f"Loaded model: {os.path.basename(file_name)}")

            self.reset_ui_state()

            if model.has_color:
                self.segment_teeth_gum_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Missing Color Data", "Model must have color information for segmentation.")
        else:
            self.statusBar().showMessage(f"Failed to load {file_name}")
            QMessageBox.critical(self, "Error", f"Failed to load {file_name}")

        self.progress_bar.setVisible(False)

    def reset_ui_state(self):
        """Reset UI state when loading new model"""
        self.step1_result_label.setText("Step 1 not completed")
        self.step2_result_label.setText("Step 2 not completed")
        self.enable_marking_btn.setText("Start Marking")
        self.enable_marking_btn.setEnabled(False)
        self.clear_current_btn.setEnabled(False)
        self.save_region_btn.setEnabled(False)
        self.cancel_region_btn.setEnabled(False)
        self.regions_list.clear()
        self.marking_enabled = False
        self.selected_region_name = None
        self.region_name_input.clear()
        self.surface_gl_widget.disable_marking()
        self.surface_gl_widget.clear_all_regions()
        self.update_region_management_buttons()
        self.current_selection_info_label.setText("")

    # Keep existing segmentation methods (run_teeth_gum_segmentation, run_surface_separation)...
    def run_teeth_gum_segmentation(self):
        """Run Step 1: Teeth/Gum segmentation"""
        if not self.model:
            return

        self.statusBar().showMessage("Running teeth/gum segmentation...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.segment_teeth_gum_btn.setEnabled(False)

        try:
            auto_refine = self.auto_refine_cb.isChecked()
            result = self.segmenter.segment_teeth_and_gums(auto_refine)

            if result:
                vertices, colors = self.segmenter.get_teeth_gum_visualization()

                if vertices is not None and colors is not None:
                    self.teeth_gum_segmented = ComparisonModel()
                    self.teeth_gum_segmented.set_data(vertices, None, colors, None, "Teeth/Gum Segmented")

                    self.teeth_gum_gl_widget.set_model(self.teeth_gum_segmented)
                    self.teeth_gum_gl_widget.show_points = True
                    self.teeth_gum_gl_widget.update()

                    teeth_points = result['teeth_points']
                    gum_points = result['gum_points']
                    other_points = result['other_points']
                    total_points = result['total_points']

                    teeth_percent = (teeth_points / total_points) * 100 if total_points > 0 else 0
                    gum_percent = (gum_points / total_points) * 100 if total_points > 0 else 0

                    self.step1_result_label.setText(
                        f"✓ Step 1 Complete\n"
                        f"Teeth: {teeth_points:,} ({teeth_percent:.1f}%)\n"
                        f"Gums: {gum_points:,} ({gum_percent:.1f}%)\n"
                        f"Other: {other_points:,}"
                    )

                    if hasattr(self.model, 'faces') and self.model.faces is not None and len(self.model.faces) > 0:
                        self.separate_surfaces_btn.setEnabled(True)
                        self.statusBar().showMessage("Step 1 complete - Ready for surface separation")
                    else:
                        self.statusBar().showMessage("Step 1 complete - No face data for surface separation")
                        QMessageBox.information(self, "No Face Data",
                                                "Model does not have face information required for surface separation.")
                else:
                    self.step1_result_label.setText("✗ Step 1 failed - visualization error")
            else:
                self.step1_result_label.setText("✗ Step 1 failed - segmentation error")

        except Exception as e:
            self.step1_result_label.setText(f"✗ Step 1 error: {str(e)}")
            print(f"Step 1 error: {e}")

        self.segment_teeth_gum_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def run_surface_separation(self):
        """Run Step 2: Surface separation on teeth"""
        if not self.segmenter.teeth_gum_result:
            return

        self.statusBar().showMessage("Running surface separation on teeth...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.separate_surfaces_btn.setEnabled(False)

        try:
            normal_threshold = self.normal_slider.value() / 10.0
            min_surface_size = self.min_size_spinbox.value()
            self.segmenter.set_surface_params(normal_threshold, min_surface_size)

            result = self.segmenter.separate_teeth_surfaces()

            if result:
                vertices, colors = self.segmenter.get_surface_separation_visualization()

                if vertices is not None and colors is not None:
                    self.surface_separated = ComparisonModel()
                    self.surface_separated.set_data(vertices, None, colors, None, "Surface Separated")

                    self.surface_gl_widget.set_model(self.surface_separated)
                    self.surface_gl_widget.show_points = True
                    self.surface_gl_widget.update()

                    inner_points = result['teeth_inner_points']
                    outer_points = result['teeth_outer_points']
                    gum_points = result['gum_points']
                    total_teeth = inner_points + outer_points

                    inner_percent = (inner_points / total_teeth) * 100 if total_teeth > 0 else 0
                    outer_percent = (outer_points / total_teeth) * 100 if total_teeth > 0 else 0

                    self.step2_result_label.setText(
                        f"✓ Step 2 Complete\n"
                        f"Inner Teeth: {inner_points:,} ({inner_percent:.1f}%)\n"
                        f"Outer Teeth: {outer_points:,} ({outer_percent:.1f}%)\n"
                        f"Gums: {gum_points:,}\n"
                        f"Total Teeth Surfaces: {total_teeth:,}"
                    )

                    self.enable_marking_btn.setEnabled(True)
                    self.statusBar().showMessage("Surface separation complete - Ready for multi-region marking")

                else:
                    self.step2_result_label.setText("✗ Step 2 failed - visualization error")
            else:
                self.step2_result_label.setText("✗ Step 2 failed - surface separation error")

        except Exception as e:
            self.step2_result_label.setText(f"✗ Step 2 error: {str(e)}")
            print(f"Step 2 error: {e}")

        self.separate_surfaces_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    # ENHANCED: New multi-region marking methods
    def toggle_marking(self):
        """Toggle marking mode on/off"""
        if not self.segmenter.surface_separation_result:
            return

        if not self.marking_enabled:
            # Enable marking
            surface_type = 'inner' if self.surface_combo.currentText() == "Inner Teeth Surface" else 'outer'
            surface_indices = self.segmenter.get_surface_indices(surface_type)

            if surface_indices is None or len(surface_indices) == 0:
                QMessageBox.warning(self, "No Surface", f"No {surface_type} teeth surface found for marking.")
                return

            vertices, colors = self.segmenter.get_surface_separation_visualization()
            if vertices is not None and colors is not None:
                self.surface_separated = ComparisonModel()
                self.surface_separated.set_data(vertices, None, colors, None, "Surface Separated")
                self.surface_gl_widget.set_model(self.surface_separated)

            self.surface_gl_widget.enable_marking(surface_type, surface_indices, self.current_brush_radius)
            self.marking_enabled = True
            self.enable_marking_btn.setText("Exit Marking")
            self.enable_marking_btn.setStyleSheet("""
                QPushButton {
                    background: linear-gradient(135deg, #e67e22, #d35400);
                    color: white;
                    padding: 10px 16px;
                    border: none;
                    border-radius: 6px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: linear-gradient(135deg, #d35400, #e67e22);
                }
            """)
            self.clear_current_btn.setEnabled(True)
            self.cancel_region_btn.setEnabled(True)

            self.selection_timer.start(1000)  # Update every second

            surface_color = "GREEN" if surface_type == "inner" else "MAGENTA"
            self.statusBar().showMessage(
                f"🎯 Multi-region marking enabled on {surface_type} teeth surface ({surface_color} areas)")

        else:
            # Disable marking
            self.surface_gl_widget.disable_marking()
            self.marking_enabled = False
            self.enable_marking_btn.setText("Start Marking")
            self.enable_marking_btn.setStyleSheet("")
            self.selection_timer.stop()

            current_count = self.surface_gl_widget.get_current_selection_count()
            if current_count > 0:
                self.statusBar().showMessage(f"Marking disabled - {current_count} points in current selection")
            else:
                self.statusBar().showMessage("Marking disabled")

    def update_brush_size(self, value):
        """Update brush size for marking"""
        actual_size = value / 100.0
        self.current_brush_radius = actual_size
        self.brush_label.setText(f"{actual_size:.2f}")

        if hasattr(self.surface_gl_widget, 'set_brush_size'):
            self.surface_gl_widget.set_brush_size(actual_size)

    def update_normal_threshold(self, value):
        """Update normal threshold for surface separation"""
        threshold = value / 10.0
        self.normal_label.setText(f"{threshold:.1f}")

    def clear_current_selection(self):
        """Clear current selection"""
        self.surface_gl_widget.clear_current_selection()
        self.region_name_input.clear()
        self.update_current_selection_info()

    def on_region_name_changed(self):
        """Handle region name input changes"""
        has_name = len(self.region_name_input.text().strip()) > 0
        has_selection = self.surface_gl_widget.get_current_selection_count() > 0
        self.save_region_btn.setEnabled(has_name and has_selection)

    def save_current_as_region(self):
        """Save current selection as a named region"""
        region_name = self.region_name_input.text().strip()

        if not region_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a region name.")
            return

        if self.surface_gl_widget.get_current_selection_count() == 0:
            QMessageBox.warning(self, "No Selection", "Please select some points first.")
            return

        # Check if region name already exists
        if region_name in self.surface_gl_widget.get_all_regions():
            reply = QMessageBox.question(self, "Region Exists",
                                         f"Region '{region_name}' already exists. Overwrite?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Save the region
        surface_type = 'inner' if self.surface_combo.currentText() == "Inner Teeth Surface" else 'outer'
        description = f"{self.surface_gl_widget.get_current_selection_count()} points on {surface_type} teeth surface"

        success = self.surface_gl_widget.save_current_selection_as_region(region_name, description)

        if success:
            self.update_regions_list()
            self.region_name_input.clear()
            self.update_current_selection_info()
            self.update_region_management_buttons()

            # Show success message
            point_count = self.surface_gl_widget.get_all_regions()[region_name]['point_count']
            self.statusBar().showMessage(f"✅ Saved region '{region_name}' with {point_count} points")
        else:
            QMessageBox.warning(self, "Save Failed", "Failed to save region. Please try again.")

    def cancel_current_region(self):
        """Cancel current region marking"""
        self.clear_current_selection()
        if self.marking_enabled:
            self.toggle_marking()

    def update_current_selection_info(self):
        """Update current selection info display"""
        if not self.marking_enabled:
            if self.surface_gl_widget.get_region_count() > 0:
                self.current_selection_info_label.setText(
                    f"📋 {self.surface_gl_widget.get_region_count()} region(s) saved\n"
                    f"🎯 Click 'Start Marking' to create new regions"
                )
            else:
                self.current_selection_info_label.setText("🎯 Click 'Start Marking' to begin selecting surface areas")
            return

        try:
            current_count = self.surface_gl_widget.get_current_selection_count()

            if current_count > 0:
                percentage = self.surface_gl_widget.get_selection_percentage()
                self.current_selection_info_label.setText(
                    f"🎯 Currently marking: {current_count:,} points\n"
                    f"📊 {percentage:.1f}% of surface selected\n"
                    f"💡 Enter name and click 'Save as Region' to preserve"
                )
            else:
                self.current_selection_info_label.setText(
                    f"🖱️ Click and drag on the surface to mark areas\n"
                    f"🎨 Selected areas will appear in bright yellow"
                )

            # Update save button state
            has_name = len(self.region_name_input.text().strip()) > 0
            self.save_region_btn.setEnabled(has_name and current_count > 0)

        except Exception as e:
            self.current_selection_info_label.setText("Marking active...")

    def update_regions_list(self):
        """Update the regions list widget with enhanced display"""
        self.regions_list.clear()

        regions = self.surface_gl_widget.get_all_regions()
        for region_name, region_data in regions.items():
            point_count = region_data['point_count']
            surface_type = region_data['surface_type']
            visible = region_data['visible']
            color = region_data['color']

            # Create display text with color and visibility indicators
            visibility_icon = "👁️" if visible else "🙈"
            color_indicator = f"●"  # Color dot

            item_text = f"{visibility_icon} {color_indicator} {region_name}"
            detail_text = f"({surface_type}, {point_count} pts)"

            item = QListWidgetItem(f"{item_text}\n{detail_text}")
            item.setData(Qt.UserRole, region_name)

            # Set color for the item
            if visible:
                item_color = QColor(color[0], color[1], color[2])
                item.setForeground(item_color)
            else:
                item.setForeground(QColor(128, 128, 128))  # Gray for hidden

            self.regions_list.addItem(item)

        self.update_region_management_buttons()

    def on_region_selected(self):
        """Handle region selection in the list"""
        current_item = self.regions_list.currentItem()
        if current_item:
            self.selected_region_name = current_item.data(Qt.UserRole)
        else:
            self.selected_region_name = None

        self.update_region_management_buttons()

    def update_region_management_buttons(self):
        """Update the state of region management buttons"""
        has_regions = self.surface_gl_widget.get_region_count() > 0
        has_selection = self.selected_region_name is not None

        self.show_hide_region_btn.setEnabled(has_selection)
        self.delete_region_btn.setEnabled(has_selection)
        self.export_selected_region_btn.setEnabled(has_selection)

        self.show_all_regions_btn.setEnabled(has_regions)
        self.hide_all_regions_btn.setEnabled(has_regions)
        self.export_all_regions_btn.setEnabled(has_regions)
        self.clear_all_regions_btn.setEnabled(has_regions)

    def toggle_selected_region_visibility(self):
        """Toggle visibility of selected region"""
        if not self.selected_region_name:
            return

        self.surface_gl_widget.toggle_region_visibility(self.selected_region_name)
        self.update_regions_list()

        region_data = self.surface_gl_widget.get_region_data(self.selected_region_name)
        if region_data:
            status = "visible" if region_data['visible'] else "hidden"
            self.statusBar().showMessage(f"Region '{self.selected_region_name}' is now {status}")

    def delete_selected_region(self):
        """Delete selected region"""
        if not self.selected_region_name:
            return

        reply = QMessageBox.question(self, "Delete Region",
                                     f"Delete region '{self.selected_region_name}'?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            success = self.surface_gl_widget.delete_region(self.selected_region_name)
            if success:
                self.update_regions_list()
                self.selected_region_name = None
                self.statusBar().showMessage(f"Deleted region '{self.selected_region_name}'")

    def show_all_regions(self):
        """Show all regions"""
        self.surface_gl_widget.show_all_regions(True)
        self.update_regions_list()
        self.statusBar().showMessage("All regions are now visible")

    def hide_all_regions(self):
        """Hide all regions"""
        self.surface_gl_widget.show_all_regions(False)
        self.update_regions_list()
        self.statusBar().showMessage("All regions are now hidden")

    def export_selected_region(self):
        """Export selected region as point cloud"""
        if not self.selected_region_name:
            QMessageBox.warning(self, "No Selection", "Please select a region first.")
            return

        region_data = self.surface_gl_widget.get_region_data(self.selected_region_name)
        if not region_data:
            QMessageBox.critical(self, "Error", f"Region '{self.selected_region_name}' not found.")
            return

        try:
            # Get model data
            vertices = self.segmenter.surface_separation_result['vertices']
            colors = self.segmenter.surface_separation_result['colors']

            # Extract point cloud
            point_cloud = self.enhanced_surface_marker.extract_region_point_cloud(
                vertices, colors, region_data['indices'], self.selected_region_name
            )

            if not point_cloud:
                QMessageBox.warning(self, "No Data", "Could not extract point cloud from region.")
                return

            # Ask for export format
            format_choice = QMessageBox.question(
                self, "Export Format",
                "Choose export format:\n\nYes = PLY file (3D compatible)\nNo = CSV file (spreadsheet compatible)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            if format_choice == QMessageBox.Cancel:
                return

            # Get save location
            if format_choice == QMessageBox.Yes:
                # PLY format
                default_filename = f"{self.selected_region_name.replace(' ', '_')}_region.ply"
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save Region as PLY", default_filename, "PLY Files (*.ply)"
                )

                if filename:
                    success = self.enhanced_surface_marker.save_point_cloud_as_ply(point_cloud, filename)
                    file_type = "PLY"
            else:
                # CSV format
                default_filename = f"{self.selected_region_name.replace(' ', '_')}_region.csv"
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save Region as CSV", default_filename, "CSV Files (*.csv)"
                )

                if filename:
                    success = self.save_point_cloud_as_csv(point_cloud, filename)
                    file_type = "CSV"

            if filename and success:
                has_colors = point_cloud['colors'] is not None
                color_info = "with RGB colors" if has_colors else "coordinates only"

                QMessageBox.information(
                    self, "Export Complete",
                    f"Region exported successfully!\n\n"
                    f"File: {os.path.basename(filename)}\n"
                    f"Format: {file_type}\n"
                    f"Points: {point_cloud['point_count']:,}\n"
                    f"Data: X,Y,Z coordinates {color_info}"
                )
                self.statusBar().showMessage(f"Region exported: {os.path.basename(filename)}")
            elif filename:
                QMessageBox.critical(self, "Export Failed", f"Failed to save region to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")

    def export_all_regions(self):
        """Export all regions as combined files"""
        regions = self.surface_gl_widget.get_all_regions()
        if not regions:
            QMessageBox.warning(self, "No Regions", "No regions to export.")
            return

        try:
            # Ask for export format
            format_choice = QMessageBox.question(
                self, "Export All Regions",
                "Choose export format:\n\nYes = Combined CSV with region labels\nNo = Individual PLY files per region",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            if format_choice == QMessageBox.Cancel:
                return

            if format_choice == QMessageBox.Yes:
                # Combined CSV export
                default_filename = "all_regions_combined.csv"
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save All Regions as Combined CSV", default_filename, "CSV Files (*.csv)"
                )

                if filename:
                    success = self.export_all_regions_as_combined_csv(filename)
                    if success:
                        QMessageBox.information(
                            self, "Export Complete",
                            f"All regions exported to combined CSV:\n{os.path.basename(filename)}\n\n"
                            f"Regions: {len(regions)}\n"
                            f"Each point includes region name and color information"
                        )
            else:
                # Individual PLY files
                folder = QFileDialog.getExistingDirectory(self, "Select Folder for PLY Files")
                if folder:
                    success_count = self.export_all_regions_as_individual_ply(folder)
                    QMessageBox.information(
                        self, "Export Complete",
                        f"Exported {success_count} out of {len(regions)} regions as individual PLY files\n"
                        f"Location: {folder}"
                    )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")

    def export_all_regions_as_combined_csv(self, filename):
        """Export all regions to a combined CSV file"""
        try:
            regions = self.surface_gl_widget.get_all_regions()
            vertices = self.segmenter.surface_separation_result['vertices']
            colors = self.segmenter.surface_separation_result['colors']

            # Prepare data for combined export
            all_regions_data = {}
            for region_name, region_data in regions.items():
                point_cloud = self.enhanced_surface_marker.extract_region_point_cloud(
                    vertices, colors, region_data['indices'], region_name
                )
                if point_cloud:
                    all_regions_data[region_name] = {
                        'point_cloud': point_cloud,
                        'color': region_data['color']
                    }

            return self.enhanced_surface_marker.save_regions_combined_csv(all_regions_data, filename)

        except Exception as e:
            print(f"Error in combined CSV export: {e}")
            return False

    def export_all_regions_as_individual_ply(self, folder):
        """Export all regions as individual PLY files"""
        try:
            regions = self.surface_gl_widget.get_all_regions()
            vertices = self.segmenter.surface_separation_result['vertices']
            colors = self.segmenter.surface_separation_result['colors']

            success_count = 0
            for region_name, region_data in regions.items():
                point_cloud = self.enhanced_surface_marker.extract_region_point_cloud(
                    vertices, colors, region_data['indices'], region_name
                )
                if point_cloud:
                    safe_name = region_name.replace(' ', '_').replace('/', '_')
                    filename = os.path.join(folder, f"{safe_name}_region.ply")
                    if self.enhanced_surface_marker.save_point_cloud_as_ply(point_cloud, filename):
                        success_count += 1

            return success_count

        except Exception as e:
            print(f"Error in individual PLY export: {e}")
            return 0

    def save_point_cloud_as_csv(self, point_cloud, filename):
        """Save point cloud to CSV format"""
        try:
            vertices = point_cloud['vertices']
            colors = point_cloud['colors']
            point_count = len(vertices)

            with open(filename, 'w') as f:
                f.write(f"# Region: {point_cloud['name']}\n")
                f.write(f"# Points: {point_count}\n")
                f.write(f"# Coordinate type: {point_cloud.get('coordinate_type', 'unknown')}\n")

                if colors is not None:
                    f.write("X,Y,Z,R,G,B\n")
                else:
                    f.write("X,Y,Z\n")

                for i in range(point_count):
                    x, y, z = vertices[i]

                    if colors is not None:
                        if np.max(colors[i]) <= 1.0:
                            r, g, b = (colors[i] * 255).astype(int)
                        else:
                            r, g, b = colors[i].astype(int)
                        f.write(f"{x:.8f},{y:.8f},{z:.8f},{r},{g},{b}\n")
                    else:
                        f.write(f"{x:.8f},{y:.8f},{z:.8f}\n")

            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False

    def clear_all_regions(self):
        """Clear all saved regions"""
        reply = QMessageBox.question(self, "Clear All Regions",
                                     "Delete all saved regions? This cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.surface_gl_widget.clear_all_regions()
            self.update_regions_list()
            self.selected_region_name = None
            self.statusBar().showMessage("All regions cleared")

    def toggle_points(self, enabled):
        """Toggle point rendering in GL widgets"""
        widgets = [self.original_gl_widget, self.teeth_gum_gl_widget, self.surface_gl_widget]

        for widget in widgets:
            if widget:
                widget.show_points = enabled
                widget.update()

        status = "enabled" if enabled else "disabled"
        self.statusBar().showMessage(f"Point rendering {status}")

    def change_point_size(self, size):
        """Change point size in GL widgets"""
        self.point_size_label.setText(str(size))

        widgets = [self.original_gl_widget, self.teeth_gum_gl_widget, self.surface_gl_widget]

        for widget in widgets:
            if widget:
                widget.set_point_size(float(size))
                widget.update()

        self.statusBar().showMessage(f"Point size set to {size}")

    def reset_view(self):
        """Reset camera view in GL widgets"""
        widgets = [self.original_gl_widget, self.teeth_gum_gl_widget, self.surface_gl_widget]

        for widget in widgets:
            if widget:
                widget.rotation_x = 0
                widget.rotation_y = 0
                widget.zoom = -3.0
                widget.update()

        self.statusBar().showMessage("View reset")

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop timers
        if hasattr(self, 'selection_timer'):
            self.selection_timer.stop()

        # Clean up GL resources
        widgets = [self.original_gl_widget, self.teeth_gum_gl_widget, self.surface_gl_widget]

        for widget in widgets:
            if widget and hasattr(widget, 'clear_model'):
                widget.clear_model()

        event.accept()


# Keep the existing IntegratedDentalSegmenter class unchanged
class IntegratedDentalSegmenter:
    """Integrated segmenter for teeth/gum segmentation + surface separation"""

    def __init__(self):
        self.model = None
        self.teeth_gum_result = None
        self.surface_separation_result = None

        # Teeth/Gum segmentation thresholds
        self.teeth_color_min = np.array([160, 160, 140])
        self.teeth_color_max = np.array([255, 255, 255])
        self.gum_color_min = np.array([150, 80, 80])
        self.gum_color_max = np.array([255, 180, 180])

        # Clustering parameters
        self.eps = 0.05
        self.min_samples = 10

        # Surface separation parameters
        self.normal_threshold = 0.4
        self.min_surface_size = 500

    def set_model(self, model):
        self.model = model
        self.teeth_gum_result = None
        self.surface_separation_result = None

    def set_surface_params(self, normal_threshold, min_surface_size):
        self.normal_threshold = normal_threshold
        self.min_surface_size = min_surface_size

    def get_surface_indices(self, surface_type):
        """Get indices for a specific surface type"""
        if not self.surface_separation_result:
            return None

        if surface_type == 'inner':
            return self.surface_separation_result['teeth_inner_indices']
        elif surface_type == 'outer':
            return self.surface_separation_result['teeth_outer_indices']
        elif surface_type == 'gums':
            return self.surface_separation_result['gum_indices']
        elif surface_type == 'other':
            return self.surface_separation_result['other_indices']
        else:
            return None

    # STEP 1: Teeth/Gum Segmentation
    def segment_teeth_and_gums(self, auto_refine=True):
        """Step 1: Segment teeth and gums"""
        if not self.model or not self.model.is_loaded():
            return None

        if not self.model.has_color:
            return None

        vertices = self.model.vertices
        colors = self.model.colors

        teeth_mask = self._create_teeth_mask(colors)
        gum_mask = self._create_gum_mask(colors)

        overlap_mask = teeth_mask & gum_mask
        gum_mask[overlap_mask] = False

        other_mask = ~(teeth_mask | gum_mask)

        teeth_indices = np.where(teeth_mask)[0]
        gum_indices = np.where(gum_mask)[0]
        other_indices = np.where(other_mask)[0]

        if auto_refine and len(teeth_indices) > 0 and len(gum_indices) > 0:
            try:
                teeth_indices, gum_indices, other_indices = self._refine_segmentation(
                    vertices, teeth_indices, gum_indices, other_indices
                )
                teeth_indices, gum_indices, other_indices = self._post_process_teeth_segmentation(
                    vertices, teeth_indices, gum_indices, other_indices
                )
            except Exception as e:
                print(f"Refinement error: {e}")

        self.teeth_gum_result = {
            'vertices': vertices,
            'colors': colors,
            'teeth_indices': teeth_indices,
            'gum_indices': gum_indices,
            'other_indices': other_indices,
            'teeth_mask': np.isin(np.arange(len(vertices)), teeth_indices),
            'gum_mask': np.isin(np.arange(len(vertices)), gum_indices),
            'other_mask': np.isin(np.arange(len(vertices)), other_indices),
            'total_points': len(vertices),
            'teeth_points': len(teeth_indices),
            'gum_points': len(gum_indices),
            'other_points': len(other_indices)
        }

        return self.teeth_gum_result

    # STEP 2: Surface Separation on Teeth
    def separate_teeth_surfaces(self):
        """Step 2: Apply surface separation only to teeth areas"""
        if not self.teeth_gum_result:
            return None

        if not self.model.faces:
            return None

        vertices = self.teeth_gum_result['vertices']
        teeth_indices = self.teeth_gum_result['teeth_indices']

        if len(teeth_indices) == 0:
            return None

        normals = self._compute_vertex_normals()
        teeth_vertices = vertices[teeth_indices]
        arch_center = self._find_arch_center(teeth_vertices)

        teeth_inner_indices, teeth_outer_indices = self._separate_teeth_surfaces(
            vertices, teeth_indices, normals, arch_center
        )

        gum_indices = self.teeth_gum_result['gum_indices']
        other_indices = self.teeth_gum_result['other_indices']

        self.surface_separation_result = {
            'vertices': vertices,
            'colors': self.teeth_gum_result['colors'],
            'teeth_inner_indices': teeth_inner_indices,
            'teeth_outer_indices': teeth_outer_indices,
            'gum_indices': gum_indices,
            'other_indices': other_indices,
            'total_points': len(vertices),
            'teeth_inner_points': len(teeth_inner_indices),
            'teeth_outer_points': len(teeth_outer_indices),
            'gum_points': len(gum_indices),
            'other_points': len(other_indices),
            'arch_center': arch_center
        }

        return self.surface_separation_result

    def _create_teeth_mask(self, colors):
        """Enhanced teeth mask to capture more teeth areas"""
        light_colored = (colors[:, 0] > 160) & (colors[:, 1] > 160) & (colors[:, 2] > 140)
        medium_light = (colors[:, 0] > 120) & (colors[:, 1] > 120) & (colors[:, 2] > 100)
        brightness = np.mean(colors, axis=1)
        bright_enough = brightness > 100
        not_too_red = (colors[:, 0] < colors[:, 1] + 50) | (colors[:, 1] > 140)
        very_red = (colors[:, 0] > colors[:, 1] + 40) & (colors[:, 0] > colors[:, 2] + 40) & (colors[:, 0] > 150)
        teeth_mask = (light_colored | (medium_light & bright_enough)) & (~very_red) & not_too_red
        return teeth_mask

    def _create_gum_mask(self, colors):
        """Create gum mask based on color"""
        red_dominant = (colors[:, 0] > colors[:, 1] + 20) & (colors[:, 0] > colors[:, 2] + 20)
        medium_red = colors[:, 0] > 120
        not_too_bright = ~((colors[:, 0] > 230) & (colors[:, 1] > 230) & (colors[:, 2] > 230))
        return red_dominant & medium_red & not_too_bright

    def _post_process_teeth_segmentation(self, vertices, teeth_indices, gum_indices, other_indices):
        """Post-process to reclaim teeth areas misclassified as 'other'"""
        if len(other_indices) == 0 or len(teeth_indices) == 0:
            return teeth_indices, gum_indices, other_indices

        kdtree = KDTree(vertices)
        other_points = vertices[other_indices]
        k = min(15, len(vertices) - 1)
        distances, neighbor_indices = kdtree.query(other_points, k=k)

        to_teeth = []

        for i, (other_idx, neighbors, dists) in enumerate(zip(other_indices, neighbor_indices, distances)):
            close_neighbors = neighbors[dists < 0.5]
            teeth_count = np.sum(np.isin(close_neighbors, teeth_indices))
            gum_count = np.sum(np.isin(close_neighbors, gum_indices))

            if teeth_count > gum_count and teeth_count > len(close_neighbors) * 0.4:
                point_color = self.model.colors[other_idx]
                if not (point_color[0] > point_color[1] + 30 and point_color[0] > 140):
                    to_teeth.append(other_idx)

        if to_teeth:
            teeth_indices = np.concatenate([teeth_indices, to_teeth])
            other_indices = np.setdiff1d(other_indices, to_teeth)

        return teeth_indices, gum_indices, other_indices

    def _refine_segmentation(self, vertices, teeth_indices, gum_indices, other_indices):
        """Refine segmentation using DBSCAN clustering"""

        def refine(indices, label):
            if len(indices) > self.min_samples:
                points = vertices[indices]
                clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
                labels = clustering.labels_
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

                if len(unique_labels) > 0:
                    max_count = counts.max()
                    significant = unique_labels[counts >= max_count * 0.2]
                    valid_mask = np.isin(labels, significant)
                    refined = indices[valid_mask]
                    removed = indices[~valid_mask]
                    return refined, removed
            return indices, []

        teeth_indices, removed_teeth = refine(teeth_indices, "Teeth")
        gum_indices, removed_gum = refine(gum_indices, "Gums")
        other_indices = np.concatenate([other_indices, removed_teeth, removed_gum])

        if len(other_indices) > 0:
            kdtree = KDTree(vertices)
            other_points = vertices[other_indices]
            k = min(20, len(vertices) - 1)

            distances, indices = kdtree.query(other_points, k=k)
            to_teeth, to_gum = [], []

            for i, neighbors in enumerate(indices):
                teeth_count = np.sum(np.isin(neighbors, teeth_indices))
                gum_count = np.sum(np.isin(neighbors, gum_indices))

                if teeth_count > gum_count and teeth_count > k * 0.5:
                    to_teeth.append(other_indices[i])
                elif gum_count > teeth_count and gum_count > k * 0.5:
                    to_gum.append(other_indices[i])

            reclassified = np.concatenate([to_teeth, to_gum])
            other_indices = np.setdiff1d(other_indices, reclassified)

            if to_teeth:
                teeth_indices = np.concatenate([teeth_indices, to_teeth])
            if to_gum:
                gum_indices = np.concatenate([gum_indices, to_gum])

        return teeth_indices, gum_indices, other_indices

    def _compute_vertex_normals(self):
        """Compute vertex normals from face data"""
        vertices = self.model.vertices
        faces = self.model.faces

        normals = np.zeros_like(vertices)

        for face in faces:
            v0, v1, v2 = vertices[face]

            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal /= norm

            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal

        norms = np.linalg.norm(normals, axis=1)
        valid_mask = norms > 0
        normals[valid_mask] = normals[valid_mask] / norms[valid_mask, np.newaxis]

        return normals

    def _find_arch_center(self, teeth_vertices):
        """Find arch center from teeth vertices"""
        if len(teeth_vertices) == 0:
            return np.array([0, 0, 0])

        min_y = np.min(teeth_vertices[:, 1])
        max_y = np.max(teeth_vertices[:, 1])
        arch_threshold = min_y + (max_y - min_y) * 0.7

        arch_mask = teeth_vertices[:, 1] > arch_threshold
        arch_vertices = teeth_vertices[arch_mask]

        if len(arch_vertices) > 0:
            return np.mean(arch_vertices, axis=0)
        else:
            return np.mean(teeth_vertices, axis=0)

    def _separate_teeth_surfaces(self, vertices, teeth_indices, normals, arch_center):
        """Separate teeth into inner and outer surfaces"""
        inner_teeth_indices = []
        outer_teeth_indices = []

        for tooth_idx in teeth_indices:
            vertex = vertices[tooth_idx]
            normal = normals[tooth_idx]

            to_vertex = vertex - arch_center
            distance = np.linalg.norm(to_vertex)

            if distance > 0:
                outward_direction = to_vertex / distance
                normal_outward_dot = np.dot(normal, outward_direction)

                if normal_outward_dot > self.normal_threshold:
                    outer_teeth_indices.append(tooth_idx)
                elif normal_outward_dot < -self.normal_threshold:
                    inner_teeth_indices.append(tooth_idx)

        if len(inner_teeth_indices) < self.min_surface_size:
            inner_teeth_indices = []
        if len(outer_teeth_indices) < self.min_surface_size:
            outer_teeth_indices = []

        return np.array(inner_teeth_indices), np.array(outer_teeth_indices)

    def get_teeth_gum_visualization(self):
        """Get visualization for teeth/gum segmentation"""
        if not self.teeth_gum_result:
            return None, None

        vertices = self.teeth_gum_result['vertices']
        colors = np.zeros_like(self.teeth_gum_result['colors'])

        colors[self.teeth_gum_result['teeth_indices']] = [255, 255, 255]
        colors[self.teeth_gum_result['gum_indices']] = [255, 100, 100]
        colors[self.teeth_gum_result['other_indices']] = [80, 80, 80]

        return vertices, colors

    def get_surface_separation_visualization(self):
        """Get visualization for surface separation results"""
        if not self.surface_separation_result:
            return None, None

        vertices = self.surface_separation_result['vertices']
        colors = np.zeros_like(self.surface_separation_result['colors'])

        colors[self.surface_separation_result['teeth_inner_indices']] = [0, 255, 0]
        colors[self.surface_separation_result['teeth_outer_indices']] = [255, 0, 255]
        colors[self.surface_separation_result['gum_indices']] = [255, 100, 100]
        colors[self.surface_separation_result['other_indices']] = [80, 80, 80]

        return vertices, colors


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Dental Segmentation with Multi-Region Marking")
    app.setOrganizationName("Dental Analysis Tools")

    configure_gl_format()

    window = EnhancedIntegratedDentalSegmentationWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()