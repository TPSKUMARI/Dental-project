#!/usr/bin/env python3
# Main window implementation with compact bottom controls

import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')  # Use Qt5 backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QCheckBox, QPushButton, QSplitter,
                             QLabel, QSlider, QAction, QFileDialog, QComboBox,
                             QDockWidget, QMessageBox, QDialog, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont

from PyQt5.QtWidgets import QCheckBox

from ui.gl_widget import GLWidget
from ui.comparison_window import ComparisonWindow
from ui.live_comparison_window import LiveComparisonWindow
from ui.comparison_controls import ComparisonControls
from utils.comparison_methods import ComparisonMethods
from utils.ply_utils import (extract_vertices_with_color, detect_tartar_by_pink_color,
                             visualize_tartar_detection, export_tartar_points,
                             visualize_point_cloud)
from ui.tartar_visualization_dialog import TartarVisualizationDialog


class PlotDialog(QDialog):
    """Dialog for displaying matplotlib figures"""

    def __init__(self, parent=None, figure=None, title="Plot Visualization"):
        super(PlotDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)

        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create canvas for the figure
        if figure:
            self.canvas = FigureCanvas(figure)
            layout.addWidget(self.canvas)

            # Add a close button
            self.close_btn = QPushButton("Close")
            self.close_btn.clicked.connect(self.accept)
            layout.addWidget(self.close_btn)


class MainWindow(QMainWindow):
    """Main application window with compact bottom controls"""

    def __init__(self):
        super(MainWindow, self).__init__()

        # Create OpenGL widgets
        self.gl_widget1 = GLWidget()
        self.gl_widget2 = GLWidget()

        # Create comparison windows
        self.comparison_window = ComparisonWindow()
        self.live_comparison_window = LiveComparisonWindow()

        # Set up live comparison callback
        self.live_comparison_window.setup_update_callback(self.update_live_comparison)

        # Create comparison controls
        self.comparison_controls = ComparisonControls()
        self.comparison_controls.parameters_changed.connect(self.on_comparison_parameters_changed)

        self.model1_loaded = False
        self.model2_loaded = False

        # For live comparison
        self.live_comparison_active = False

        # Create timer for performance optimization
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.perform_comparison)

        # Connect signals
        self.gl_widget1.file_loaded.connect(lambda: self.on_file_loaded(1))
        self.gl_widget2.file_loaded.connect(lambda: self.on_file_loaded(2))

        # Connect view change signals for synchronized rotation and zoom
        self.gl_widget1.view_changed.connect(self.sync_view_from_widget1)
        self.gl_widget2.view_changed.connect(self.sync_view_from_widget2)

        self.initUI()

    def initUI(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(8)  # Reduced spacing
        main_layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins

        # Create compact title section
        title_frame = QFrame()
        title_frame.setFrameStyle(QFrame.Box)
        title_frame.setStyleSheet("""
            QFrame {
                background-color: #3B3B3B;
                border: 2px solid #B28228;
                border-radius: 6px;
                padding: 3px;
            }
        """)
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(5, 5, 5, 5)

        title_label = QLabel("Dental Tartar Analysis System")
        title_font = QFont()
        title_font.setPointSize(14)  # Smaller font
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #B28228; background-color: transparent; border: none;")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)

        main_layout.addWidget(title_frame)

        # Create compact model labels section
        labels_layout = QHBoxLayout()
        labels_layout.setSpacing(5)

        model1_label = QLabel("Model 1: Before Tablet")
        model1_label.setAlignment(Qt.AlignCenter)
        model1_label.setStyleSheet("""
            QLabel {
                background-color: #3B3B3B;
                border: 2px solid #B28228;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                color: #B28228;
                font-size: 11px;
            }
        """)

        model2_label = QLabel("Model 2: After Tablet (Pink = Tartar)")
        model2_label.setAlignment(Qt.AlignCenter)
        model2_label.setStyleSheet("""
            QLabel {
                background-color: #3B3B3B;
                border: 2px solid #B28228;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                color: #B28228;
                font-size: 11px;
            }
        """)

        labels_layout.addWidget(model1_label)
        labels_layout.addWidget(model2_label)
        main_layout.addLayout(labels_layout)

        # Create splitter for side-by-side view - This gets most of the space
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.gl_widget1)
        splitter.addWidget(self.gl_widget2)
        splitter.setSizes([400, 400])
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #B28228;
                width: 4px;
                border-radius: 2px;
            }
        """)

        # Add splitter with high stretch factor to take most space
        main_layout.addWidget(splitter, 10)  # High stretch factor

        # Create compact control panel
        control_panel = QGroupBox("Options")
        control_panel.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                color: #B28228;
                border: 2px solid #B28228;
                border-radius: 6px;
                margin-top: 1ex;
                padding-top: 8px;
                background-color: #3B3B3B;
                max-height: 80px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px 0 6px;
                color: #B28228;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(8, 5, 8, 5)

        # Compact display options
        display_controls = QHBoxLayout()
        display_controls.setSpacing(8)

        self.wireframe_cb = QCheckBox("Wireframe")
        self.wireframe_cb.toggled.connect(self.toggle_wireframe)
        display_controls.addWidget(self.wireframe_cb)

        self.points_cb = QCheckBox("Points")
        self.points_cb.toggled.connect(self.toggle_points)
        display_controls.addWidget(self.points_cb)

        self.vbo_cb = QCheckBox("GPU")
        self.vbo_cb.setChecked(True)
        self.vbo_cb.toggled.connect(self.toggle_vbo)
        display_controls.addWidget(self.vbo_cb)

        control_layout.addLayout(display_controls)

        # Add compact separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #B28228;")
        control_layout.addWidget(separator)

        # Compact analysis options
        analysis_controls = QHBoxLayout()
        analysis_controls.setSpacing(8)

        self.sync_views_cb = QCheckBox("Sync Views")
        self.sync_views_cb.setChecked(True)
        analysis_controls.addWidget(self.sync_views_cb)

        self.live_compare_cb = QCheckBox("Live Compare")
        self.live_compare_cb.toggled.connect(self.toggle_live_comparison)
        analysis_controls.addWidget(self.live_compare_cb)

        control_layout.addLayout(analysis_controls)

        # Add compact separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("color: #B28228;")
        control_layout.addWidget(separator2)

        # Compact compare button
        self.compare_btn = QPushButton("🔍 Compare")
        self.compare_btn.clicked.connect(self.compare_models)
        self.compare_btn.setEnabled(False)
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B3B3B;
                border: 2px solid #B28228;
                border-radius: 6px;
                padding: 6px 12px;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #B28228;
                color: #1C1C1C;
            }
            QPushButton:pressed {
                background-color: #E5A300;
                color: #1C1C1C;
            }
            QPushButton:disabled {
                background-color: #2A2A2A;
                border-color: #555555;
                color: #777777;
            }
        """)
        control_layout.addWidget(self.compare_btn)

        # Add control panel with minimal stretch
        main_layout.addWidget(control_panel, 0)  # No stretch

        # Create VERY compact comparison controls section
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #1C1C1C;
                border: 2px solid #B28228;
                border-radius: 6px;
                padding: 5px;
            }
        """)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(3, 3, 3, 3)
        controls_layout.setSpacing(3)

        # Compact controls title
        controls_title = QLabel("Tartar Detection")
        controls_title.setStyleSheet("""
            QLabel {
                color: #B28228;
                font-weight: bold;
                font-size: 12px;
                background-color: transparent;
                border: none;
                padding: 2px 0px;
            }
        """)
        controls_title.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(controls_title)

        # Make the comparison controls more compact
        self.comparison_controls.setMaximumHeight(120)  # Limit height
        controls_layout.addWidget(self.comparison_controls)

        # Add controls frame with minimal stretch
        main_layout.addWidget(controls_frame, 0)  # No stretch

        # Set central widget
        self.setCentralWidget(main_widget)

        # Compact status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready. Open PLY files to begin analysis.")
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #3B3B3B;
                color: #FFFFFF;
                border-top: 2px solid #B28228;
                font-weight: bold;
                padding: 3px;
                font-size: 11px;
            }
        """)

        # Create menu
        self.create_menu()

        # Set window properties
        self.setWindowTitle("Dental Tartar Analysis System")
        self.resize(1200, 800)  # Back to reasonable size

    def create_menu(self):
        # File menu
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #3B3B3B;
                color: #FFFFFF;
                border: none;
                padding: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
                border-radius: 4px;
                margin: 1px;
            }
            QMenuBar::item:selected {
                background-color: #B28228;
                color: #1C1C1C;
            }
        """)

        file_menu = menubar.addMenu("📁 File")

        # Open Model 1
        open_model1_action = QAction("📂 Open Model 1 (Before Tablet)", self)
        open_model1_action.setShortcut("Ctrl+1")
        open_model1_action.triggered.connect(lambda: self.open_file(1))
        file_menu.addAction(open_model1_action)

        # Open Model 2
        open_model2_action = QAction("📂 Open Model 2 (After Tablet)", self)
        open_model2_action.setShortcut("Ctrl+2")
        open_model2_action.triggered.connect(lambda: self.open_file(2))
        file_menu.addAction(open_model2_action)

        file_menu.addSeparator()

        # Export options
        export_menu = file_menu.addMenu("💾 Export")

        # Export vertices with color from Model 1
        export_model1_action = QAction("📊 Export Model 1 Vertices with Color", self)
        export_model1_action.triggered.connect(lambda: self.export_vertices(1))
        export_menu.addAction(export_model1_action)

        # Export vertices with color from Model 2
        export_model2_action = QAction("📊 Export Model 2 Vertices with Color", self)
        export_model2_action.triggered.connect(lambda: self.export_vertices(2))
        export_menu.addAction(export_model2_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("❌ Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Compare menu
        compare_menu = menubar.addMenu("🔍 Compare")

        # Static Compare action
        static_compare_action = QAction("📸 Static Comparison", self)
        static_compare_action.setShortcut("Ctrl+C")
        static_compare_action.triggered.connect(self.compare_models)
        compare_menu.addAction(static_compare_action)

        # Live Compare action
        live_compare_action = QAction("🎥 Live Comparison", self)
        live_compare_action.setShortcut("Ctrl+L")
        live_compare_action.setCheckable(True)
        live_compare_action.toggled.connect(self.live_compare_cb.setChecked)
        compare_menu.addAction(live_compare_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("🦷 Analysis")

        # Dental tartar analysis
        dental_analysis_action = QAction("🔬 Analyze Dental Tartar", self)
        dental_analysis_action.triggered.connect(self.analyze_dental_models)
        analysis_menu.addAction(dental_analysis_action)

        # Export tartar points
        export_tartar_action = QAction("📋 Export Tartar Points", self)
        export_tartar_action.triggered.connect(self.export_tartar_points)
        analysis_menu.addAction(export_tartar_action)

    def open_file(self, model_num):
        """Open a PLY file for a specific model"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Open PLY File for Model {model_num}", "", "PLY Files (*.ply)"
        )

        if file_name:
            if model_num == 1:
                if self.gl_widget1.load_ply(file_name):
                    info = self.gl_widget1.get_model_info()
                    self.status_bar.showMessage(
                        f"Model 1: {os.path.basename(file_name)} - Vertices: {info['vertices']}")
                else:
                    self.status_bar.showMessage(f"Failed to load {file_name}")
            else:
                if self.gl_widget2.load_ply(file_name):
                    info = self.gl_widget2.get_model_info()
                    self.status_bar.showMessage(
                        f"Model 2: {os.path.basename(file_name)} - Vertices: {info['vertices']}")
                else:
                    self.status_bar.showMessage(f"Failed to load {file_name}")

    def export_vertices(self, model_num):
        """Export vertices with color from the specified model"""
        # Check if model is loaded
        if (model_num == 1 and not self.model1_loaded) or (model_num == 2 and not self.model2_loaded):
            self.status_bar.showMessage(f"Model {model_num} not loaded.")
            return

        # Get the filename
        if model_num == 1:
            ply_file = self.gl_widget1.current_file
        else:
            ply_file = self.gl_widget2.current_file

        if not ply_file:
            self.status_bar.showMessage(f"No file for Model {model_num}.")
            return

        # Get save location
        save_path, _ = QFileDialog.getSaveFileName(
            self, f"Save Vertices from Model {model_num}", "", "CSV Files (*.csv)")

        if not save_path:
            return

        # Extract and save the vertices
        try:
            vertices, colors, has_color = extract_vertices_with_color(ply_file)

            if vertices is not None:
                if has_color:
                    data = np.column_stack((vertices, colors))
                    header = "x,y,z,red,green,blue"
                    np.savetxt(save_path, data, delimiter=',', header=header, comments='',
                               fmt='%.6f,%.6f,%.6f,%d,%d,%d')
                    self.status_bar.showMessage(f"💾 Exported {len(vertices)} vertices from Model {model_num}")
                else:
                    np.savetxt(save_path, vertices, delimiter=',', header="x,y,z", comments='', fmt='%.6f,%.6f,%.6f')
                    self.status_bar.showMessage(f"💾 Exported {len(vertices)} vertices from Model {model_num}")
            else:
                self.status_bar.showMessage(f"Failed to extract vertices from Model {model_num}.")

        except Exception as e:
            self.status_bar.showMessage(f"Error exporting vertices: {e}")

    def analyze_dental_models(self):
        """Analyze dental models to detect tartar areas - FIXED to ensure correct model assignment"""
        if not (self.model1_loaded and self.model2_loaded):
            self.status_bar.showMessage("⚠️ Both models must be loaded for analysis.")
            return

        # IMPORTANT: Ensure we're using the correct files for each model
        # Model 1 = Before Tablet (clean teeth) - should be loaded in gl_widget1
        # Model 2 = After Tablet (with pink tartar) - should be loaded in gl_widget2
        before_tablet_file = self.gl_widget1.current_file  # Model 1 - Before tablet
        after_tablet_file = self.gl_widget2.current_file  # Model 2 - After tablet

        print(f"DEBUG: Before tablet file (Model 1): {before_tablet_file}")
        print(f"DEBUG: After tablet file (Model 2): {after_tablet_file}")

        self.status_bar.showMessage("🔬 Starting tartar analysis...")

        try:
            # Load Model 1 (Before Tablet - Clean Teeth)
            print("DEBUG: Loading Model 1 (Before Tablet)")
            vertices_before, colors_before, has_color_before = extract_vertices_with_color(before_tablet_file)
            if vertices_before is None:
                self.status_bar.showMessage("❌ Failed to load Model 1 (Before Tablet)")
                return

            print(f"DEBUG: Model 1 loaded - {len(vertices_before)} vertices")

            # Load Model 2 (After Tablet - With Pink Tartar)
            print("DEBUG: Loading Model 2 (After Tablet)")
            vertices_after, colors_after, has_color_after = extract_vertices_with_color(after_tablet_file)
            if vertices_after is None:
                self.status_bar.showMessage("❌ Failed to load Model 2 (After Tablet)")
                return

            print(f"DEBUG: Model 2 loaded - {len(vertices_after)} vertices")

            if not (has_color_before and has_color_after):
                self.status_bar.showMessage("⚠️ Both models must have color information!")
                return

            # Store original data for verification
            original_vertices_before = vertices_before.copy()
            original_colors_before = colors_before.copy()
            original_vertices_after = vertices_after.copy()
            original_colors_after = colors_after.copy()

            # Align models if they have different vertex counts
            if len(vertices_before) != len(vertices_after):
                print(f"DEBUG: Aligning models - Before: {len(vertices_before)}, After: {len(vertices_after)}")
                try:
                    from scipy.spatial import cKDTree
                    tree_after = cKDTree(vertices_after)
                    distances, indices = tree_after.query(vertices_before, k=1)

                    # Align the AFTER model to match the BEFORE model structure
                    # This ensures we can compare corresponding points
                    colors_after = colors_after[indices]
                    vertices_after = vertices_after[indices]  # Also align vertices for consistency

                    print("DEBUG: Models aligned successfully")
                    self.status_bar.showMessage("✅ Models aligned successfully.")
                except ImportError:
                    self.status_bar.showMessage("⚠️ SciPy required for model alignment")
                    return
                except Exception as e:
                    print(f"DEBUG: Alignment error: {e}")
                    return

            # Detect tartar areas
            print("DEBUG: Detecting tartar areas...")
            tartar_mask = detect_tartar_by_pink_color(colors_before, colors_after)
            if tartar_mask is None:
                self.status_bar.showMessage("❌ Failed to detect tartar areas.")
                return

            tartar_count = np.sum(tartar_mask)
            if tartar_count == 0:
                self.status_bar.showMessage("ℹ️ No tartar areas detected.")
                return

            tartar_percentage = (tartar_count / len(vertices_before)) * 100
            print(f"DEBUG: Detected {tartar_count} tartar points ({tartar_percentage:.2f}%)")

            # Show visualization with CORRECT model assignment
            print("DEBUG: Creating visualization dialog...")
            visualization_dialog = TartarVisualizationDialog(self,
                                                             title=f"🦷 Tartar Analysis - {tartar_percentage:.2f}% Detected")

            # CRITICAL: Pass the data in the correct order
            # vertices_before, colors_before = Model 1 (Before Tablet - Clean)
            # original_vertices_after, original_colors_after = Model 2 (After Tablet - Pink)
            # tartar_mask = detected on the aligned data
            print("DEBUG: Setting visualization data...")
            visualization_dialog.set_data(
                vertices_before,  # Model 1 vertices (before tablet)
                colors_before,  # Model 1 colors (before tablet)
                original_vertices_after,  # Model 2 vertices (after tablet) - ORIGINAL data
                original_colors_after,  # Model 2 colors (after tablet) - ORIGINAL data
                tartar_mask  # Tartar detection mask
            )

            # Show the dialog
            visualization_dialog.exec_()

            self.status_bar.showMessage(f"🎉 Detected {tartar_count} tartar points ({tartar_percentage:.2f}%)")

        except Exception as e:
            print(f"DEBUG: Analysis error: {str(e)}")
            self.status_bar.showMessage(f"❌ Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()

    def export_tartar_points(self):
        """Export tartar points from dental models"""
        if not (self.model1_loaded and self.model2_loaded):
            self.status_bar.showMessage("Both models must be loaded.")
            return

        try:
            before_tablet_file = self.gl_widget1.current_file
            after_tablet_file = self.gl_widget2.current_file

            vertices_before, colors_before, has_color_before = extract_vertices_with_color(before_tablet_file)
            vertices_after, colors_after, has_color_after = extract_vertices_with_color(after_tablet_file)

            if not (has_color_before and has_color_after):
                self.status_bar.showMessage("Both models must have color information.")
                return

            # Align if needed
            if len(vertices_before) != len(vertices_after):
                try:
                    from scipy.spatial import cKDTree
                    tree_after = cKDTree(vertices_after)
                    distances, indices = tree_after.query(vertices_before, k=1)
                    colors_after = colors_after[indices]
                except ImportError:
                    self.status_bar.showMessage("SciPy required for alignment")
                    return

            tartar_mask = detect_tartar_by_pink_color(colors_before, colors_after)
            if tartar_mask is None or np.sum(tartar_mask) == 0:
                self.status_bar.showMessage("No tartar areas to export.")
                return

            save_path, _ = QFileDialog.getSaveFileName(self, "Save Tartar Points", "", "CSV Files (*.csv)")
            if not save_path:
                return

            tartar_vertices = vertices_before[tartar_mask]
            tartar_colors = colors_after[tartar_mask]
            data = np.column_stack((tartar_vertices, tartar_colors))

            np.savetxt(save_path, data, delimiter=',', header="x,y,z,red,green,blue", comments='',
                       fmt='%.6f,%.6f,%.6f,%d,%d,%d')
            self.status_bar.showMessage(f"💾 Exported {len(tartar_vertices)} tartar points")

        except Exception as e:
            self.status_bar.showMessage(f"Error exporting: {str(e)}")

    def on_file_loaded(self, model_num):
        """Handle model loaded signal"""
        if model_num == 1:
            self.model1_loaded = True
        else:
            self.model2_loaded = True

        if self.model1_loaded and self.model2_loaded:
            self.compare_btn.setEnabled(True)
            self.status_bar.showMessage("Both models loaded. Ready for comparison.")

            if self.sync_views_cb.isChecked():
                self.sync_views()

            if self.live_compare_cb.isChecked():
                self.start_live_comparison()

    def toggle_wireframe(self, checked):
        """Toggle wireframe rendering mode"""
        self.gl_widget1.show_wireframe = checked
        self.gl_widget2.show_wireframe = checked
        self.gl_widget1.update()
        self.gl_widget2.update()

    def toggle_points(self, checked):
        """Toggle point rendering mode"""
        self.gl_widget1.show_points = checked
        self.gl_widget2.show_points = checked
        self.gl_widget1.update()
        self.gl_widget2.update()

    def toggle_vbo(self, checked):
        """Toggle VBO (GPU acceleration) usage"""
        self.gl_widget1.use_vbo = checked
        self.gl_widget2.use_vbo = checked
        self.gl_widget1.update()
        self.gl_widget2.update()

    def toggle_live_comparison(self, checked):
        """Toggle live comparison mode"""
        if checked and self.model1_loaded and self.model2_loaded:
            self.start_live_comparison()
        else:
            self.stop_live_comparison()

    def sync_views(self):
        """Synchronize views between both models"""
        self.gl_widget2.set_view(
            self.gl_widget1.rotation_x,
            self.gl_widget1.rotation_y,
            self.gl_widget1.zoom
        )

    def sync_view_from_widget1(self, rotation_x, rotation_y, zoom):
        """Sync view from widget 1 to widget 2"""
        if self.sync_views_cb.isChecked():
            self.gl_widget2.set_view(rotation_x, rotation_y, zoom)
            if self.live_comparison_active:
                self.update_timer.start(50)

    def sync_view_from_widget2(self, rotation_x, rotation_y, zoom):
        """Sync view from widget 2 to widget 1"""
        if self.sync_views_cb.isChecked():
            self.gl_widget1.set_view(rotation_x, rotation_y, zoom)
            if self.live_comparison_active:
                self.update_timer.start(50)

    def compare_models(self):
        """Compare the two models"""
        if not (self.model1_loaded and self.model2_loaded):
            self.status_bar.showMessage("Load both models first")
            return

        if self.sync_views_cb.isChecked():
            self.sync_views()
        self.perform_comparison(static=True)

    def start_live_comparison(self):
        """Start live comparison updates"""
        if not (self.model1_loaded and self.model2_loaded):
            self.live_compare_cb.setChecked(False)
            return

        self.live_comparison_active = True
        self.update_live_comparison()
        self.live_comparison_window.show()
        self.live_comparison_window.start_live_updates(200)

    def stop_live_comparison(self):
        """Stop live comparison updates"""
        self.live_comparison_active = False
        self.live_comparison_window.stop_live_updates()
        self.live_comparison_window.hide()

    def update_live_comparison(self):
        """Update the live comparison"""
        if self.live_comparison_active and self.model1_loaded and self.model2_loaded:
            self.perform_comparison(static=False)

    def on_comparison_parameters_changed(self, method, threshold):
        """Handle comparison parameter changes"""
        if self.live_comparison_active:
            self.update_live_comparison()

    def perform_comparison(self, static=False):
        """Perform the actual comparison"""
        img1 = self.gl_widget1.render_to_image(512, 512)
        img2 = self.gl_widget2.render_to_image(512, 512)

        width = min(img1.width(), img2.width())
        height = min(img1.height(), img2.height())

        method, threshold = self.comparison_controls.get_current_parameters()
        diff_img, tartar_count, total_pixels = ComparisonMethods.detect_dental_tartar(img1, img2, width, height,
                                                                                      threshold)

        diff_percentage = (tartar_count / total_pixels) * 100
        status_msg = f"Found {tartar_count} tartar pixels ({diff_percentage:.2f}%)"
        self.status_bar.showMessage(status_msg)

        file1 = self.gl_widget1.current_file if self.gl_widget1.current_file else "Model 1"
        file2 = self.gl_widget2.current_file if self.gl_widget2.current_file else "Model 2"

        stats = (f"<b>Dental Tartar Analysis</b><br>"
                 f"<b>Sensitivity:</b> {threshold}<br>"
                 f"<b>Before Tablet:</b> {os.path.basename(file1) if file1 != 'Model 1' else file1}<br>"
                 f"<b>After Tablet:</b> {os.path.basename(file2) if file2 != 'Model 2' else file2}<br>"
                 f"<b>Tartar Pixels:</b> {tartar_count} out of {total_pixels} ({diff_percentage:.2f}%)<br>"
                 f"<b>Color Legend:</b><br>"
                 f"<font color='red'>■</font> Tartar Areas<br>"
                 f"<font color='blue'>■</font> Other Differences<br>"
                 f"<font color='green'>■</font> Matching Areas")

        if static:
            self.comparison_window.show_comparison(img1, img2, diff_img, stats)
        else:
            self.live_comparison_window.update_comparison(img1, img2, diff_img, stats)

    def closeEvent(self, event):
        """Handle window close event"""
        self.comparison_window.close()
        self.live_comparison_window.close()
        event.accept()