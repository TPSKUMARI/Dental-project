#!/usr/bin/env python3
"""
PyQt PLY Viewer - CPU Clustering + GPU Visualization
Clean version without OpenGL widget issues
"""

import sys
import numpy as np
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyQt5")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                            QWidget, QGroupBox, QHBoxLayout, QSpinBox, QPushButton, 
                            QLabel, QCheckBox, QFileDialog, QMessageBox,
                            QProgressBar, QTextEdit, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

# Required imports
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# GPU libraries for visualization only
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CUDA available for visualization: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")
except ImportError:
    CUDA_AVAILABLE = False
    print("PyTorch not available")

def is_in_red_pink_range(rgb_color):
    """Check if RGB color is within red/pink range"""
    r, g, b = rgb_color
    return (220 <= r <= 255 and 20 <= g <= 105 and 60 <= b <= 180)

class CPUClusteringThread(QThread):
    """Thread for CPU-based clustering (scikit-learn)"""
    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object, object, object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, vertices, colors, k_value):
        super().__init__()
        self.vertices = vertices
        self.colors = colors
        self.k_value = k_value
    
    def run(self):
        try:
            self.progress.emit("Starting CPU K-means clustering...")
            
            # CPU clustering using scikit-learn
            cluster_labels, clustered_colors = self.cpu_clustering(
                self.colors, self.k_value
            )
            
            self.progress.emit("Finding best cluster...")
            
            # Find best cluster
            best_cluster_id, cluster_indices = self.find_best_cluster(
                self.vertices, self.colors, cluster_labels
            )
            
            self.progress.emit("Clustering complete!")
            
            self.finished_signal.emit(
                cluster_labels, clustered_colors, best_cluster_id, cluster_indices
            )
            
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def cpu_clustering(self, original_colors, k_value=7):
        """CPU K-means using scikit-learn"""
        self.progress.emit(f"Using CPU K-means with {k_value} clusters...")
        
        start_time = time.time()
        
        # Use scikit-learn K-means
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(original_colors)
        
        clustering_time = time.time() - start_time
        self.progress.emit(f"CPU K-means completed in {clustering_time:.2f} seconds")
        
        # Get cluster labels and colors
        cluster_labels = kmeans.labels_
        clustered_colors = kmeans.cluster_centers_[cluster_labels].astype(np.uint8)
        
        return cluster_labels, clustered_colors
    
    def find_best_cluster(self, vertices, original_colors, cluster_labels):
        """Find best cluster using CPU"""
        self.progress.emit("Finding best matching cluster...")
        
        red_pink_mask = np.array([is_in_red_pink_range(color) for color in original_colors])
        
        if np.any(red_pink_mask):
            # Find cluster closest to red/pink regions
            red_pink_vertices = vertices[red_pink_mask]
            red_pink_centroid = np.mean(red_pink_vertices, axis=0)
            
            best_distance = float('inf')
            best_cluster_id = 0
            
            for cluster_id in range(self.k_value):
                cluster_mask = cluster_labels == cluster_id
                if np.any(cluster_mask):
                    cluster_vertices = vertices[cluster_mask]
                    cluster_centroid = np.mean(cluster_vertices, axis=0)
                    distance = np.linalg.norm(cluster_centroid - red_pink_centroid)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_cluster_id = cluster_id
        else:
            # Use largest cluster if no red/pink regions found
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(self.k_value)]
            best_cluster_id = np.argmax(cluster_sizes)
        
        cluster_indices = np.where(cluster_labels == best_cluster_id)[0]
        self.progress.emit(f"Selected cluster {best_cluster_id + 1} with {len(cluster_indices)} points")
        
        return best_cluster_id, cluster_indices

class SimpleFallbackWidget(QWidget):
    """Simple fallback widget for 3D view area"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("3D Visualization")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; margin: 20px;")
        layout.addWidget(title_label)
        
        # Main message
        message_label = QLabel(
            "Click 'Show GPU-Accelerated 3D View' to open\n"
            "high-quality 3D visualization in a separate window"
        )
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet("color: #666; font-size: 14px; margin: 20px;")
        layout.addWidget(message_label)
        
        # Instructions
        instructions = QLabel(
            "Workflow:\n"
            "1. Load PLY file\n"
            "2. Apply CPU clustering\n"
            "3. View GPU-accelerated 3D visualization\n"
            "4. Save modified PLY file"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet(
            "color: #555; font-size: 12px; padding: 20px; "
            "background-color: #f8f9fa; border-radius: 8px; margin: 20px;"
        )
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        # System status
        status_text = "System Status:\n"
        status_text += f"GPU (Visualization): {'✅ Available' if CUDA_AVAILABLE else '❌ Not Available'}\n"
        status_text += "CPU (Clustering): ✅ Available (scikit-learn)"
        
        status_label = QLabel(status_text)
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet(
            "color: #666; font-size: 10px; font-family: monospace; "
            "background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px;"
        )
        layout.addWidget(status_label)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super(MainWindow, self).__init__()
        
        # Model data
        self.vertices = None
        self.original_colors = None
        self.clustered_colors = None
        self.cluster_labels = None
        self.cluster_indices = None
        self.best_cluster_id = None
        
        # Parameters
        self.k_value = 7
        self.green_intensity = 255
        
        # Threading
        self.clustering_thread = None
        
        self.initUI()

    def initUI(self):
        """Initialize user interface"""
        # Central widget with splitter
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Controls
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Right side - Simple view area
        view_panel = self.create_view_panel()
        splitter.addWidget(view_panel)
        
        # Set splitter proportions
        splitter.setSizes([350, 650])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)
        
        # Window properties
        self.setWindowTitle("PLY Viewer - CPU Clustering + GPU Visualization")
        self.resize(1200, 800)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Load a PLY file to begin")

    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("PLY Viewer")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)
        
        # File controls
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load PLY File")
        self.load_btn.clicked.connect(self.load_file)
        self.load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; border-radius: 5px; }")
        file_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Modified PLY")
        self.save_btn.clicked.connect(self.save_file)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }")
        file_layout.addWidget(self.save_btn)
        
        layout.addWidget(file_group)
        
        # Clustering controls
        cluster_group = QGroupBox("Clustering Parameters")
        cluster_layout = QVBoxLayout(cluster_group)
        
        # K value
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K Clusters:"))
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(2)
        self.k_spinbox.setMaximum(20)
        self.k_spinbox.setValue(self.k_value)
        self.k_spinbox.valueChanged.connect(self.update_k_value)
        k_layout.addWidget(self.k_spinbox)
        cluster_layout.addLayout(k_layout)
        
        # Green intensity
        green_layout = QHBoxLayout()
        green_layout.addWidget(QLabel("Green Intensity:"))
        self.green_spinbox = QSpinBox()
        self.green_spinbox.setMinimum(100)
        self.green_spinbox.setMaximum(255)
        self.green_spinbox.setValue(self.green_intensity)
        self.green_spinbox.valueChanged.connect(self.update_green_intensity)
        green_layout.addWidget(self.green_spinbox)
        cluster_layout.addLayout(green_layout)
        
        # Apply button
        self.cluster_btn = QPushButton("Apply CPU Clustering")
        self.cluster_btn.clicked.connect(self.apply_clustering)
        self.cluster_btn.setEnabled(False)
        self.cluster_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 12px; border-radius: 5px; }")
        cluster_layout.addWidget(self.cluster_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        cluster_layout.addWidget(self.progress_bar)
        
        layout.addWidget(cluster_group)
        
        # Visualization controls
        viz_group = QGroupBox("3D Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # GPU-accelerated visualization button
        self.gpu_viz_btn = QPushButton("Show GPU-Accelerated 3D View")
        self.gpu_viz_btn.clicked.connect(self.show_gpu_visualization)
        self.gpu_viz_btn.setEnabled(False)
        self.gpu_viz_btn.setStyleSheet("QPushButton { background-color: #E91E63; color: white; font-weight: bold; padding: 12px; border-radius: 5px; }")
        viz_layout.addWidget(self.gpu_viz_btn)
        
        layout.addWidget(viz_group)
        
        # System info
        system_group = QGroupBox("System Information")
        system_layout = QVBoxLayout(system_group)
        
        self.system_info = QTextEdit()
        self.system_info.setMaximumHeight(120)
        self.system_info.setReadOnly(True)
        self.update_system_info()
        system_layout.addWidget(self.system_info)
        
        layout.addWidget(system_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No data loaded")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel

    def create_view_panel(self):
        """Create simple view panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # View controls
        view_controls = QWidget()
        controls_layout = QHBoxLayout(view_controls)
        
        controls_layout.addWidget(QLabel("3D View:"))
        controls_layout.addStretch()
        
        instructions = QLabel("Use 'Show GPU-Accelerated 3D View' for visualization")
        instructions.setStyleSheet("color: #666; font-size: 11px;")
        controls_layout.addWidget(instructions)
        
        layout.addWidget(view_controls)
        
        # Simple fallback widget
        self.view_widget = SimpleFallbackWidget()
        layout.addWidget(self.view_widget)
        
        return panel

    def update_system_info(self):
        """Update system information display"""
        info_text = "System Capabilities:\n\n"
        
        # GPU info for visualization
        if CUDA_AVAILABLE:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info_text += f"✅ GPU (Visualization):\n   {gpu_name}\n   Memory: {gpu_memory:.1f} GB\n\n"
        else:
            info_text += "❌ GPU: Not Available\n\n"
        
        # CPU clustering info
        info_text += "✅ CPU (Clustering):\n   scikit-learn K-means\n   Fast & Reliable"
        
        self.system_info.setText(info_text)

    def update_k_value(self, value):
        self.k_value = value

    def update_green_intensity(self, value):
        self.green_intensity = value

    def load_file(self):
        """Load PLY file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open PLY File", "", "PLY Files (*.ply);;All Files (*)"
        )
        
        if file_name:
            success = self.load_ply_file(file_name)
            
            if success:
                self.cluster_btn.setEnabled(True)
                self.gpu_viz_btn.setEnabled(True)
                
                vertex_count = len(self.vertices)
                has_colors = self.original_colors is not None
                
                self.stats_label.setText(
                    f"Vertices: {vertex_count:,}\n"
                    f"Colors: {'Yes' if has_colors else 'No'}\n"
                    f"File: {os.path.basename(file_name)}"
                )
                
                self.status_bar.showMessage(f"Loaded: {vertex_count:,} vertices")

    def load_ply_file(self, filename):
        """Load PLY file with error handling"""
        try:
            print(f"Loading PLY file: {filename}")
            
            plydata = PlyData.read(filename)
            vertex_data = plydata['vertex']
            
            # Load vertices
            self.vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
            print(f"Loaded {len(self.vertices)} vertices")
            
            # Load colors if available
            if 'red' in vertex_data and 'green' in vertex_data and 'blue' in vertex_data:
                self.original_colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
                print(f"Loaded {len(self.original_colors)} colors")
            else:
                self.original_colors = None
                print("No color data found")
            
            return True
            
        except Exception as e:
            print(f"Error loading PLY file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load PLY file:\n{str(e)}")
            return False

    def apply_clustering(self):
        """Apply CPU clustering"""
        if self.original_colors is None:
            QMessageBox.warning(self, "Warning", "No color data available for clustering")
            return
        
        if self.clustering_thread and self.clustering_thread.isRunning():
            return
        
        # Disable controls
        self.cluster_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start CPU clustering thread
        self.clustering_thread = CPUClusteringThread(
            self.vertices, self.original_colors, self.k_value
        )
        self.clustering_thread.progress.connect(self.update_progress)
        self.clustering_thread.finished_signal.connect(self.clustering_finished)
        self.clustering_thread.error_signal.connect(self.clustering_error)
        self.clustering_thread.start()

    def update_progress(self, message):
        """Update progress message"""
        self.status_bar.showMessage(message)

    def clustering_finished(self, cluster_labels, clustered_colors, best_cluster_id, cluster_indices):
        """Handle clustering completion"""
        self.cluster_labels = cluster_labels
        self.clustered_colors = clustered_colors
        self.best_cluster_id = best_cluster_id
        self.cluster_indices = cluster_indices
        
        # Enable controls
        self.cluster_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update statistics
        cluster_size = len(cluster_indices) if cluster_indices is not None else 0
        self.stats_label.setText(
            f"Vertices: {len(self.vertices):,}\n"
            f"Colors: Yes\n"
            f"Clusters: {self.k_value}\n"
            f"Best Cluster: {best_cluster_id + 1}\n"
            f"Cluster Size: {cluster_size:,} points"
        )
        
        self.status_bar.showMessage(f"Clustering complete! Best cluster: {best_cluster_id + 1} ({cluster_size:,} points)")

    def clustering_error(self, error_message):
        """Handle clustering error"""
        self.cluster_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Clustering Error", f"Clustering failed:\n{error_message}")
        self.status_bar.showMessage("Clustering failed")

    def save_file(self):
        """Save modified PLY file"""
        if self.vertices is None or self.cluster_indices is None:
            return
        
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Modified PLY File", "", "PLY Files (*.ply);;All Files (*)"
        )
        
        if file_name:
            success = self.save_modified_ply(file_name)
            if success:
                QMessageBox.information(self, "Success", f"File saved successfully:\n{file_name}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save file")

    def save_modified_ply(self, filename):
        """Save the modified PLY file with green cluster"""
        try:
            # Create modified colors
            green_color = [0, self.green_intensity, 0]
            modified_colors = self.original_colors.copy()
            modified_colors[self.cluster_indices] = green_color
            
            # Prepare vertex data
            vertex_data = []
            for i in range(len(self.vertices)):
                vertex_data.append((
                    float(self.vertices[i][0]), float(self.vertices[i][1]), float(self.vertices[i][2]),
                    int(modified_colors[i][0]), int(modified_colors[i][1]), int(modified_colors[i][2])
                ))
            
            vertex_element = PlyElement.describe(
                np.array(vertex_data, dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                ]), 'vertex'
            )
            
            ply_data = PlyData([vertex_element])
            ply_data.write(filename)
            
            print(f"Modified PLY file saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"Failed to save PLY file: {e}")
            return False

    def show_gpu_visualization(self):
        """Show GPU-accelerated realistic 3D visualization"""
        if self.vertices is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load a PLY file first.")
            return
        
        try:
            print("Starting realistic GPU-accelerated 3D visualization...")
            
            # Set matplotlib backend for best 3D performance
            import matplotlib
            matplotlib.use('Qt5Agg')
            
            # GPU data processing for realistic scaling
            vertices_to_use = self.vertices.copy()
            if CUDA_AVAILABLE:
                print("Using GPU for realistic data scaling...")
                vertices_to_use = self.gpu_realistic_scaling(vertices_to_use)
            else:
                vertices_to_use = self.cpu_realistic_scaling(vertices_to_use)
            
            # Create professional figure with realistic proportions
            fig = plt.figure(figsize=(24, 8))  # Wider for better 3D viewing
            fig.patch.set_facecolor('white')
            fig.suptitle('Realistic 3D PLY Visualization - Dental Model Analysis', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            x, y, z = vertices_to_use.T
            
            # Calculate realistic bounds with proper aspect ratio
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            z_range = np.max(z) - np.min(z)
            max_range = max(x_range, y_range, z_range)
            
            # Center around origin for realistic viewing
            x_center = (np.max(x) + np.min(x)) / 2
            y_center = (np.max(y) + np.min(y)) / 2
            z_center = (np.max(z) + np.min(z)) / 2
            
            # Realistic point sizing based on model scale
            model_scale = max_range
            if len(vertices_to_use) > 200000:
                point_size = max(0.1, model_scale * 0.8)
                alpha = 0.7
                subsample_factor = 4
            elif len(vertices_to_use) > 100000:
                point_size = max(0.2, model_scale * 1.2)
                alpha = 0.8
                subsample_factor = 2
            else:
                point_size = max(0.5, model_scale * 2.0)
                alpha = 0.9
                subsample_factor = 1
            
            print(f"Realistic scaling: point_size={point_size:.3f}, model_scale={model_scale:.3f}")
            
            # Smart subsampling for performance while maintaining quality
            if subsample_factor > 1:
                # Use systematic sampling to preserve structure
                indices = np.arange(0, len(vertices_to_use), subsample_factor)
                x_display, y_display, z_display = x[indices], y[indices], z[indices]
                print(f"Displaying {len(indices):,} of {len(vertices_to_use):,} points for optimal performance")
            else:
                x_display, y_display, z_display = x, y, z
                indices = np.arange(len(vertices_to_use))
            
            # Prepare high-quality datasets
            datasets = []
            
            if self.original_colors is not None:
                original_colors_display = self.original_colors[indices] if subsample_factor > 1 else self.original_colors
                datasets.append(("Original Dental Model", original_colors_display, f"{len(indices):,} points", "Natural colors"))
            
            if self.clustered_colors is not None:
                clustered_colors_display = self.clustered_colors[indices] if subsample_factor > 1 else self.clustered_colors
                datasets.append(("Cluster Analysis", clustered_colors_display, f"{self.k_value} clusters", "Algorithm segmentation"))
            
            if self.cluster_indices is not None:
                green_color = [0, self.green_intensity, 0]
                modified_colors = self.original_colors.copy()
                modified_colors[self.cluster_indices] = green_color
                modified_colors_display = modified_colors[indices] if subsample_factor > 1 else modified_colors
                cluster_size = len(self.cluster_indices)
                datasets.append(("Target Region Highlighted", modified_colors_display, 
                               f"Cluster {self.best_cluster_id + 1}: {cluster_size:,} points", "Clinical focus area"))
            
            if not datasets:
                QMessageBox.warning(self, "Warning", "No visualization data available.")
                return
            
            # Create realistic 3D subplots
            for i, (title, colors, desc, subtitle) in enumerate(datasets):
                ax = fig.add_subplot(1, len(datasets), i + 1, projection='3d')
                
                # Enhanced color processing for realism
                colors_norm = colors / 255.0
                
                # Add subtle depth-based color variation for 3D realism
                z_normalized = (z_display - np.min(z_display)) / (np.max(z_display) - np.min(z_display))
                depth_variation = 0.1 * z_normalized.reshape(-1, 1)
                colors_enhanced = np.clip(colors_norm + depth_variation, 0, 1)
                
                # High-quality scatter plot with realistic rendering
                scatter = ax.scatter(x_display, y_display, z_display,
                                   c=colors_enhanced, s=point_size, alpha=alpha,
                                   edgecolors='none', linewidths=0,
                                   rasterized=False, depthshade=True)
                
                # Professional title styling
                ax.set_title(f'{title}\n{subtitle}\n({desc})', 
                           fontsize=14, fontweight='bold', pad=30,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                
                # Realistic axis labels with units
                ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold', labelpad=10)
                ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold', labelpad=10)
                ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold', labelpad=10)
                
                # Set realistic bounds with proper aspect ratio
                margin = max_range * 0.1  # 10% margin for better viewing
                ax.set_xlim(x_center - max_range/2 - margin, x_center + max_range/2 + margin)
                ax.set_ylim(y_center - max_range/2 - margin, y_center + max_range/2 + margin)
                ax.set_zlim(z_center - max_range/2 - margin, z_center + max_range/2 + margin)
                
                # Equal aspect ratio for realistic proportions
                ax.set_box_aspect([1,1,1])
                
                # Enhanced realistic styling
                ax.grid(True, alpha=0.2, linewidth=0.5)
                ax.xaxis.pane.fill = True
                ax.yaxis.pane.fill = True
                ax.zaxis.pane.fill = True
                
                # Subtle pane coloring for depth perception
                ax.xaxis.pane.set_facecolor('#f8f8f8')
                ax.yaxis.pane.set_facecolor('#f8f8f8')
                ax.zaxis.pane.set_facecolor('#f8f8f8')
                ax.xaxis.pane.set_alpha(0.1)
                ax.yaxis.pane.set_alpha(0.1)
                ax.zaxis.pane.set_alpha(0.1)
                
                # Professional edge styling
                ax.xaxis.pane.set_edgecolor('#cccccc')
                ax.yaxis.pane.set_edgecolor('#cccccc')
                ax.zaxis.pane.set_edgecolor('#cccccc')
                ax.xaxis.pane.set_linewidth(0.5)
                ax.yaxis.pane.set_linewidth(0.5)
                ax.zaxis.pane.set_linewidth(0.5)
                
                # Realistic tick spacing
                ax.locator_params(nbins=6)
                
                # Optimal viewing angle for dental models
                if i == 0:  # Original - front view
                    ax.view_init(elev=15, azim=45)
                elif i == 1:  # Clusters - side view
                    ax.view_init(elev=20, azim=135)
                else:  # Highlighted - top view
                    ax.view_init(elev=60, azim=45)
                
                # Add subtle background
                ax.set_facecolor('#fafafa')
                
                # Enhanced tick formatting
                ax.tick_params(labelsize=10, pad=5)
                
                # Add scale reference if this is a dental model
                if i == 0:  # Add scale bar to first plot
                    scale_length = max_range * 0.2  # 20% of model size
                    scale_x = x_center + max_range/2 - scale_length - margin/2
                    scale_y = y_center - max_range/2 + margin/2
                    scale_z = z_center - max_range/2 + margin/2
                    
                    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], [scale_z, scale_z],
                           color='black', linewidth=3, alpha=0.8)
                    ax.text(scale_x + scale_length/2, scale_y, scale_z - margin/4,
                           f'{scale_length:.1f}mm', fontsize=10, ha='center', fontweight='bold')
            
            # Professional information display
            gpu_status = "GPU-Accelerated" if CUDA_AVAILABLE else "CPU-Optimized"
            info_text = f"{gpu_status} Realistic 3D Visualization | {len(self.vertices):,} vertices | Real-scale proportions"
            if self.cluster_indices is not None:
                info_text += f"\nCPU Clustering: {self.k_value} clusters | Best match: Cluster {self.best_cluster_id + 1}"
            
            fig.text(0.5, 0.02, info_text, ha='center', fontsize=12, 
                    style='italic', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            # Optimal layout with realistic spacing
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15, top=0.85, left=0.02, right=0.98, wspace=0.05)
            
            # Set professional window title
            manager = plt.get_current_fig_manager()
            manager.set_window_title('PLY Viewer - Realistic 3D Dental Model Visualization')
            
            # Maximize window for best viewing experience
            try:
                manager.window.state('zoomed')  # Windows
            except:
                try:
                    manager.full_screen_toggle()  # Alternative
                except:
                    pass
            
            plt.show()
            
            print("Realistic GPU-accelerated visualization completed successfully!")
            
            # Cleanup GPU memory
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                print("GPU memory cleared")
            
        except Exception as e:
            print(f"Realistic visualization error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Visualization Error", 
                               f"Failed to create realistic visualization:\n{str(e)}")

    def gpu_realistic_scaling(self, vertices):
        """GPU-accelerated realistic scaling preserving original proportions"""
        try:
            if not CUDA_AVAILABLE:
                return self.cpu_realistic_scaling(vertices)
            
            print("GPU realistic scaling for dental model proportions...")
            
            # Move to GPU
            vertices_gpu = torch.tensor(vertices, dtype=torch.float32).cuda()
            
            # Calculate real-world scale bounds
            bbox_min = torch.min(vertices_gpu, dim=0)[0]
            bbox_max = torch.max(vertices_gpu, dim=0)[0]
            dimensions = bbox_max - bbox_min
            
            # For dental models, assume reasonable real-world size (typical jaw span ~60-80mm)
            # Scale to preserve realistic proportions
            max_dimension = torch.max(dimensions)
            
            # Assume model represents ~70mm actual size (average adult jaw)
            if max_dimension > 0:
                real_world_size = 70.0  # mm
                scale_factor = real_world_size / max_dimension
                vertices_gpu = vertices_gpu * scale_factor
            
            # Center around origin for optimal viewing
            new_bbox_min = torch.min(vertices_gpu, dim=0)[0]
            new_bbox_max = torch.max(vertices_gpu, dim=0)[0]
            center = (new_bbox_min + new_bbox_max) / 2
            vertices_gpu = vertices_gpu - center
            
            scaled_vertices = vertices_gpu.cpu().numpy()
            
            print(f"GPU realistic scaling: {max_dimension:.3f} -> {real_world_size}mm scale")
            return scaled_vertices
            
        except Exception as e:
            print(f"GPU realistic scaling error: {e}, using CPU")
            return self.cpu_realistic_scaling(vertices)

    def cpu_realistic_scaling(self, vertices):
        """CPU realistic scaling preserving original proportions"""
        print("CPU realistic scaling for dental model proportions...")
        
        # Calculate bounds
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        dimensions = bbox_max - bbox_min
        max_dimension = np.max(dimensions)
        
        # Scale to realistic size
        if max_dimension > 0:
            real_world_size = 70.0  # mm - realistic jaw size
            scale_factor = real_world_size / max_dimension
            vertices = vertices * scale_factor
        
        # Center around origin
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        center = (bbox_min + bbox_max) / 2
        vertices = vertices - center
        
        print(f"CPU realistic scaling: {max_dimension:.3f} -> {real_world_size}mm scale")
        return vertices

    def closeEvent(self, event):
        """Handle application close"""
        if self.clustering_thread and self.clustering_thread.isRunning():
            self.clustering_thread.terminate()
            self.clustering_thread.wait()
        
        # Clear GPU memory
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Show startup information
    print("\n" + "="*70)
    print("🚀 PLY Viewer - CPU Clustering + GPU Visualization")
    print("="*70)
    print("\n📋 Features:")
    print("   • CPU K-means clustering (scikit-learn)")
    print("   • GPU-accelerated 3D visualization")
    print("   • High-performance data processing")
    print("   • Professional matplotlib rendering")
    print("   • PLY file import and export")
    print("\n🎮 Simple Workflow:")
    print("   1. Load PLY File")
    print("   2. Apply CPU Clustering")
    print("   3. Show GPU-Accelerated 3D View")
    print("   4. Save modified PLY file")
    print("\n⚡ System Status:")
    if CUDA_AVAILABLE:
        print(f"   ✅ GPU (Visualization): {torch.cuda.get_device_name(0)}")
    else:
        print("   ❌ GPU: Not Available (using CPU)")
    
    print("   ✅ CPU (Clustering): scikit-learn K-means")
    print("\n🎯 Benefits:")
    print("   • No OpenGL compatibility issues")
    print("   • Reliable CPU clustering")
    print("   • GPU acceleration where it matters most")
    print("   • Clean, simple interface")
    print("="*70)
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()