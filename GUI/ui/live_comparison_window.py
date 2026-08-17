#!/usr/bin/env python3
# Window for real-time comparison during model manipulation

from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer


class LiveComparisonWindow(QWidget):
    """Window for displaying real-time comparison results"""

    def __init__(self, parent=None):
        super(LiveComparisonWindow, self).__init__()
        self.setWindowTitle("Live Comparison Results")
        self.resize(1200, 600)

        # Create main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Images layout
        image_layout = QHBoxLayout()

        # Create labels for the images
        self.model1_label = QLabel("Model 1")
        self.model1_label.setAlignment(Qt.AlignCenter)
        self.model1_label.setMinimumSize(350, 350)

        self.model2_label = QLabel("Model 2")
        self.model2_label.setAlignment(Qt.AlignCenter)
        self.model2_label.setMinimumSize(350, 350)

        self.diff_label = QLabel("Difference")
        self.diff_label.setAlignment(Qt.AlignCenter)
        self.diff_label.setMinimumSize(350, 350)

        # Add labels to layout
        image_layout.addWidget(self.model1_label)
        image_layout.addWidget(self.model2_label)
        image_layout.addWidget(self.diff_label)
        main_layout.addLayout(image_layout)

        # Statistics layout
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self.stats_label)
        main_layout.addLayout(stats_layout)

        # Create timer for updating the comparison
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.request_update)

        # Store the callback function
        self.update_callback = None

    def setup_update_callback(self, callback_function):
        """Set the callback function for requesting updates"""
        self.update_callback = callback_function

    def request_update(self):
        """Request a new comparison update from the main window"""
        if self.update_callback:
            self.update_callback()

    def start_live_updates(self, interval=100):
        """Start live updates with the specified interval (in ms)"""
        self.update_timer.start(interval)

    def stop_live_updates(self):
        """Stop live updates"""
        self.update_timer.stop()

    def update_comparison(self, img1, img2, diff_img, stats=None):
        """Update the comparison visualization"""
        self.model1_label.setPixmap(QPixmap.fromImage(img1).scaled(
            self.model1_label.width(), self.model1_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.model2_label.setPixmap(QPixmap.fromImage(img2).scaled(
            self.model2_label.width(), self.model2_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.diff_label.setPixmap(QPixmap.fromImage(diff_img).scaled(
            self.diff_label.width(), self.diff_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Update statistics if provided
        if stats:
            self.stats_label.setText(stats)

    def closeEvent(self, event):
        """Handle window close event to stop the timer"""
        self.stop_live_updates()
        event.accept()