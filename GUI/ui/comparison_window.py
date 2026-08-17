#!/usr/bin/env python3
# Comparison window for showing side-by-side models and differences

from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt

class ComparisonWindow(QWidget):
    """Window for displaying comparison results"""

    def __init__(self, parent=None):
        super(ComparisonWindow, self).__init__()
        self.setWindowTitle("Pixel-wise Comparison Results")
        self.resize(1200, 600)

        # Create layout
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

        # Color legend
        legend_layout = QHBoxLayout()

        # Colors vary based on comparison method
        legend_layout.addWidget(QLabel("Color Legend:"))

        # Add color boxes with labels
        legend_layout.addWidget(self.create_color_sample(QColor(0, 255, 0), "Match"))
        legend_layout.addWidget(self.create_color_sample(QColor(255, 255, 0), "Close match"))
        legend_layout.addWidget(self.create_color_sample(QColor(255, 0, 0), "Different"))
        legend_layout.addWidget(self.create_color_sample(QColor(255, 0, 255), "Red channel difference"))
        legend_layout.addWidget(self.create_color_sample(QColor(0, 255, 255), "Green channel difference"))
        legend_layout.addWidget(self.create_color_sample(QColor(255, 255, 0), "Blue channel difference"))

        main_layout.addLayout(legend_layout)

    def create_color_sample(self, color, text):
        """Create a label with color sample and description"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        color_sample = QLabel()
        color_sample.setFixedSize(16, 16)
        color_sample.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid black;")

        layout.addWidget(color_sample)
        layout.addWidget(QLabel(text))

        return widget

    def update_tartar_legend(self):
        """Update the color legend for tartar detection mode"""
        # Clear the existing legend
        for i in reversed(range(self.legend_layout.count())):
            self.legend_layout.itemAt(i).widget().setParent(None)

        # Add tartar-specific legend
        self.legend_layout.addWidget(QLabel("Tartar Analysis Legend:"))
        self.legend_layout.addWidget(self.create_color_sample(QColor(255, 0, 0), "Definite Tartar"))
        self.legend_layout.addWidget(self.create_color_sample(QColor(255, 192, 203), "Possible Tartar"))
        self.legend_layout.addWidget(self.create_color_sample(QColor(0, 0, 255), "Other Differences"))
        self.legend_layout.addWidget(self.create_color_sample(QColor(0, 255, 0), "Matching Areas"))

    def show_comparison(self, img1, img2, diff_img, stats=None):
        """Set the images for comparison"""
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

        self.show()