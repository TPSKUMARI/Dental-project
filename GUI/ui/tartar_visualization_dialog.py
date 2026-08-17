#!/usr/bin/env python3
# Dialog for visualizing dental tartar using OpenGL point cloud rendering
# Fixed to show correct models in correct panels

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt
from ui.point_cloud_viewer import PointCloudViewer
import numpy as np


class TartarVisualizationDialog(QDialog):
    """Dialog for visualizing dental tartar with fast OpenGL rendering - Fixed version"""

    def __init__(self, parent=None, title="Dental Tartar Visualization"):
        super(TartarVisualizationDialog, self).__init__(parent)

        # Set dialog properties
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)

        # Create the layout
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Create viewer widgets
        self.viewers_layout = QHBoxLayout()

        # Create three viewers: before, after, highlighted
        self.viewer_before = PointCloudViewer(self)
        self.viewer_after = PointCloudViewer(self)
        self.viewer_highlighted = PointCloudViewer(self)

        # Connect mouse move events for synced camera movement
        self.viewer_before.mouseMoveEvent = self.create_synced_mouse_move(self.viewer_before)
        self.viewer_after.mouseMoveEvent = self.create_synced_mouse_move(self.viewer_after)
        self.viewer_highlighted.mouseMoveEvent = self.create_synced_mouse_move(self.viewer_highlighted)

        # Connect wheel events for synced zooming
        self.viewer_before.wheelEvent = self.create_synced_wheel_event(self.viewer_before)
        self.viewer_after.wheelEvent = self.create_synced_wheel_event(self.viewer_after)
        self.viewer_highlighted.wheelEvent = self.create_synced_wheel_event(self.viewer_highlighted)

        # Add labels with clear descriptions
        before_container = QVBoxLayout()
        before_label = QLabel("Before Tablet")
        before_label.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 12px;")
        before_label.setAlignment(Qt.AlignCenter)
        before_container.addWidget(before_label)
        before_container.addWidget(self.viewer_before)

        after_container = QVBoxLayout()
        after_label = QLabel("After Tablet (Pink = Tartar)")
        after_label.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 12px;")
        after_label.setAlignment(Qt.AlignCenter)
        after_container.addWidget(after_label)
        after_container.addWidget(self.viewer_after)

        highlight_container = QVBoxLayout()
        highlight_label = QLabel("Tartar Highlighted")
        highlight_label.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 12px;")
        highlight_label.setAlignment(Qt.AlignCenter)
        highlight_container.addWidget(highlight_label)
        highlight_container.addWidget(self.viewer_highlighted)

        # Add containers to the layout
        self.viewers_layout.addLayout(before_container)
        self.viewers_layout.addLayout(after_container)
        self.viewers_layout.addLayout(highlight_container)

        main_layout.addLayout(self.viewers_layout)

        # Add statistics label
        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 14px; color: #FFFFFF; font-weight: bold;")
        main_layout.addWidget(self.stats_label)

        # Add controls
        controls_layout = QHBoxLayout()

        # Point size control
        point_size_group = QGroupBox("Point Size")
        point_size_layout = QVBoxLayout(point_size_group)
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(2)
        self.point_size_slider.valueChanged.connect(self.change_point_size)
        point_size_layout.addWidget(self.point_size_slider)
        controls_layout.addWidget(point_size_group)

        # Sync views checkbox
        self.sync_views_checkbox = QCheckBox("Sync Camera Views")
        self.sync_views_checkbox.setChecked(True)
        controls_layout.addWidget(self.sync_views_checkbox)

        # Reset view button
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_views)
        controls_layout.addWidget(self.reset_view_button)

        # Export button
        self.export_button = QPushButton("Export Tartar Points")
        controls_layout.addWidget(self.export_button)

        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        controls_layout.addWidget(self.close_button)

        main_layout.addLayout(controls_layout)

        # Add instructions
        instructions = QLabel(
            "Controls: Left-click drag = Rotate | Right-click drag = Pan | Scroll = Zoom | R = Reset View | +/- = Adjust Point Size"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: #CCCCCC; font-size: 10px;")
        main_layout.addWidget(instructions)

    def set_data(self, vertices_before, colors_before, vertices_after, colors_after, tartar_mask):
        """Set the data for visualization - FIXED to show correct models"""
        print("DEBUG: Setting data for tartar visualization")
        print(f"DEBUG: Before model vertices: {len(vertices_before)}")
        print(f"DEBUG: After model vertices: {len(vertices_after)}")
        print(f"DEBUG: Tartar points detected: {np.sum(tartar_mask)}")

        # IMPORTANT: Make sure we're using the correct models for each viewer
        # vertices_before and colors_before are from Model 1 (before tablet - clean teeth)
        # vertices_after and colors_after are from Model 2 (after tablet - with pink tartar)

        # Set data for BEFORE viewer (Model 1 - clean teeth)
        print("DEBUG: Setting before viewer with Model 1 data (clean teeth)")
        self.viewer_before.set_data(vertices_before, colors_before)

        # Set data for AFTER viewer (Model 2 - with pink tartar areas)
        print("DEBUG: Setting after viewer with Model 2 data (pink tartar)")
        self.viewer_after.set_data(vertices_after, colors_after)

        # Set data for HIGHLIGHTED viewer (use Model 2 vertices with tartar mask highlighting)
        print("DEBUG: Setting highlighted viewer with Model 2 data + tartar mask")
        self.viewer_highlighted.set_data(vertices_after, colors_after, tartar_mask)

        # Update statistics
        tartar_count = np.sum(tartar_mask.astype(bool))
        total_points = len(vertices_before)
        tartar_percentage = (tartar_count / total_points) * 100

        self.stats_label.setText(
            f"Detected {tartar_count} tartar points out of {total_points} total points ({tartar_percentage:.2f}%)"
        )

        # Connect export button to the parent's export function
        if hasattr(self.parent(), "export_tartar_points"):
            self.export_button.clicked.connect(self.parent().export_tartar_points)

        print("DEBUG: Tartar visualization data set successfully")

    def create_synced_mouse_move(self, source_viewer):
        """Create a mouse move event handler that syncs all views if sync is enabled"""
        original_handler = source_viewer.mouseMoveEvent

        def synced_mouse_move(event):
            # Call the original handler first
            original_handler(event)

            # If sync is enabled, update other viewers
            if self.sync_views_checkbox.isChecked():
                # Get camera parameters from the source viewer
                rotation_x = source_viewer.rotation_x
                rotation_y = source_viewer.rotation_y
                translation = source_viewer.translation.copy()
                scale = source_viewer.scale

                # Update other viewers
                for viewer in [self.viewer_before, self.viewer_after, self.viewer_highlighted]:
                    if viewer != source_viewer:
                        viewer.rotation_x = rotation_x
                        viewer.rotation_y = rotation_y
                        viewer.translation = translation.copy()
                        viewer.scale = scale
                        viewer.update()

        return synced_mouse_move

    def create_synced_wheel_event(self, source_viewer):
        """Create a wheel event handler that syncs all views if sync is enabled"""
        original_handler = source_viewer.wheelEvent

        def synced_wheel_event(event):
            # Call the original handler first
            original_handler(event)

            # If sync is enabled, update other viewers
            if self.sync_views_checkbox.isChecked():
                # Get scale from the source viewer
                scale = source_viewer.scale

                # Update other viewers
                for viewer in [self.viewer_before, self.viewer_after, self.viewer_highlighted]:
                    if viewer != source_viewer:
                        viewer.scale = scale
                        viewer.update()

        return synced_wheel_event

    def change_point_size(self, size):
        """Change the point size in all viewers"""
        self.viewer_before.point_size = size
        self.viewer_after.point_size = size
        self.viewer_highlighted.point_size = size

        self.viewer_before.update()
        self.viewer_after.update()
        self.viewer_highlighted.update()

    def reset_views(self):
        """Reset all viewers to the default view"""
        self.viewer_before.reset_view()
        self.viewer_after.reset_view()
        self.viewer_highlighted.reset_view()