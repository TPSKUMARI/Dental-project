#!/usr/bin/env python3
# Compact controls for adjusting comparison parameters
# Modified to be much smaller and take less vertical space

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class ComparisonControls(QWidget):
    """Compact widget for controlling comparison parameters"""

    parameters_changed = pyqtSignal(str, int)  # method_name, threshold_value

    def __init__(self, parent=None):
        super(ComparisonControls, self).__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background-color: #1C1C1C;
                color: #FFFFFF;
            }
        """)
        self.init_ui()

    def init_ui(self):
        """Initialize compact UI components"""
        # Main horizontal layout to save vertical space
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 5, 10, 5)

        # Method selection section - compact
        method_section = QVBoxLayout()
        method_section.setSpacing(3)

        method_title = QLabel("Method:")
        method_title.setStyleSheet("""
            QLabel {
                color: #B28228;
                font-weight: bold;
                font-size: 11px;
                background-color: transparent;
                border: none;
            }
        """)
        method_section.addWidget(method_title)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Dental Tartar Detection"])
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        self.method_combo.setStyleSheet("""
            QComboBox {
                background-color: #3B3B3B;
                border: 2px solid #B28228;
                border-radius: 4px;
                padding: 4px 8px;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 10px;
                min-width: 120px;
                max-height: 25px;
            }
            QComboBox:hover {
                border-color: #FFFFFF;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
                background-color: #B28228;
                border-radius: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #3B3B3B;
                border: 2px solid #B28228;
                border-radius: 4px;
                color: #FFFFFF;
                selection-background-color: #B28228;
                selection-color: #1C1C1C;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        method_section.addWidget(self.method_combo)
        main_layout.addLayout(method_section)

        # Compact sensitivity slider section
        sensitivity_section = QVBoxLayout()
        sensitivity_section.setSpacing(3)

        sensitivity_title = QLabel("Sensitivity:")
        sensitivity_title.setStyleSheet("""
            QLabel {
                color: #B28228;
                font-weight: bold;
                font-size: 11px;
                background-color: transparent;
                border: none;
            }
        """)
        sensitivity_section.addWidget(sensitivity_title)

        # Horizontal layout for slider and labels
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(8)

        # Min label
        min_label = QLabel("5")
        min_label.setStyleSheet("""
            QLabel {
                color: #AAAAAA;
                font-size: 9px;
                background-color: transparent;
                border: none;
                min-width: 15px;
            }
        """)
        slider_layout.addWidget(min_label)

        # Slider
        self.tartar_slider = QSlider(Qt.Horizontal)
        self.tartar_slider.setRange(5, 50)
        self.tartar_slider.setValue(20)
        self.tartar_slider.valueChanged.connect(lambda v: self.on_slider_changed("Dental Tartar Detection", v))
        self.tartar_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3B3B3B;
                height: 8px;
                background: #2A2A2A;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #B28228;
                border: 1px solid #B28228;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #FFFFFF;
                border: 1px solid #FFFFFF;
            }
            QSlider::sub-page:horizontal {
                background: #B28228;
                border-radius: 4px;
            }
        """)
        self.tartar_slider.setMinimumWidth(150)
        self.tartar_slider.setMaximumHeight(20)
        slider_layout.addWidget(self.tartar_slider)

        # Max label
        max_label = QLabel("50")
        max_label.setStyleSheet("""
            QLabel {
                color: #AAAAAA;
                font-size: 9px;
                background-color: transparent;
                border: none;
                min-width: 15px;
            }
        """)
        slider_layout.addWidget(max_label)

        sensitivity_section.addLayout(slider_layout)
        main_layout.addLayout(sensitivity_section)

        # Current value display - compact
        value_section = QVBoxLayout()
        value_section.setSpacing(3)

        value_title = QLabel("Value:")
        value_title.setStyleSheet("""
            QLabel {
                color: #B28228;
                font-weight: bold;
                font-size: 11px;
                background-color: transparent;
                border: none;
            }
        """)
        value_section.addWidget(value_title)

        self.tartar_value_label = QLabel("20")
        self.tartar_value_label.setStyleSheet("""
            QLabel {
                color: #1C1C1C;
                font-size: 14px;
                font-weight: bold;
                background-color: #B28228;
                border: 2px solid #B28228;
                border-radius: 4px;
                padding: 3px 6px;
                min-width: 25px;
                max-height: 25px;
            }
        """)
        self.tartar_value_label.setAlignment(Qt.AlignCenter)
        value_section.addWidget(self.tartar_value_label)
        main_layout.addLayout(value_section)

        # Tips section - very compact
        tips_section = QVBoxLayout()
        tips_section.setSpacing(2)

        tips_title = QLabel("Tips:")
        tips_title.setStyleSheet("""
            QLabel {
                color: #B28228;
                font-weight: bold;
                font-size: 11px;
                background-color: transparent;
                border: none;
            }
        """)
        tips_section.addWidget(tips_title)

        tips_label = QLabel("↑ Higher = More Sensitive\n↓ Lower = Less Sensitive")
        tips_label.setStyleSheet("""
            QLabel {
                color: #CCCCCC;
                font-size: 9px;
                background-color: #2A2A2A;
                border: 1px solid #B28228;
                border-radius: 3px;
                padding: 3px;
                max-height: 35px;
            }
        """)
        tips_label.setWordWrap(True)
        tips_section.addWidget(tips_label)
        main_layout.addLayout(tips_section)

        # Set initial state
        self.on_method_changed(0)

    def on_method_changed(self, index):
        """Handle method selection change"""
        method_name = self.method_combo.currentText()
        self.parameters_changed.emit(method_name, self.tartar_slider.value())

    def on_slider_changed(self, slider_name, value):
        """Handle slider value changes with compact visual feedback"""
        self.tartar_value_label.setText(str(value))

        # Change color based on sensitivity level - more compact styling
        if value <= 15:
            # Low sensitivity - blue
            color_style = """
                QLabel {
                    color: #FFFFFF;
                    font-size: 14px;
                    font-weight: bold;
                    background-color: #4A90E2;
                    border: 2px solid #4A90E2;
                    border-radius: 4px;
                    padding: 3px 6px;
                    min-width: 25px;
                    max-height: 25px;
                }
            """
        elif value <= 35:
            # Medium sensitivity - gold
            color_style = """
                QLabel {
                    color: #1C1C1C;
                    font-size: 14px;
                    font-weight: bold;
                    background-color: #B28228;
                    border: 2px solid #B28228;
                    border-radius: 4px;
                    padding: 3px 6px;
                    min-width: 25px;
                    max-height: 25px;
                }
            """
        else:
            # High sensitivity - red
            color_style = """
                QLabel {
                    color: #FFFFFF;
                    font-size: 14px;
                    font-weight: bold;
                    background-color: #E74C3C;
                    border: 2px solid #E74C3C;
                    border-radius: 4px;
                    padding: 3px 6px;
                    min-width: 25px;
                    max-height: 25px;
                }
            """

        self.tartar_value_label.setStyleSheet(color_style)

        # Emit the change signal
        current_method = self.method_combo.currentText()
        self.parameters_changed.emit(current_method, value)

    def get_current_parameters(self):
        """Get the currently selected parameters"""
        method = self.method_combo.currentText()
        return method, self.tartar_slider.value()