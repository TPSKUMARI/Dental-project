"""
UI styling and theme for the PLY Viewer application
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt


# Application color scheme based on the provided palette
class Colors:
    """Color constants used throughout the application"""

    # Main colors from the provided palette
    LIGHT_BEIGE = "#E6E5E0"
    TAN = "#D0C09E"
    GOLD = "#DBBB5F"
    DARK_GOLD = "#DFB011"
    BLACK = "#000000"

    # Dark theme UI colors - inverted from original
    BACKGROUND = BLACK
    WIDGET_BACKGROUND = "#1A1A1A"  # Darker black for widgets
    TEXT = LIGHT_BEIGE
    HIGHLIGHT = DARK_GOLD
    BUTTON = GOLD
    BUTTON_TEXT = BLACK
    DISABLED = "#666666"
    ACCENT = DARK_GOLD

    # Transparency variants
    TRANSPARENT_GOLD = "rgba(219, 187, 95, 180)"

    # Helper methods to convert HEX to RGB float (for OpenGL)
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert hex color string to RGB tuple with values from 0-1"""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    @staticmethod
    def hex_to_rgba(hex_color, alpha=1.0):
        """Convert hex color string to RGBA tuple with values from 0-1"""
        rgb = Colors.hex_to_rgb(hex_color)
        return (rgb[0], rgb[1], rgb[2], alpha)

    @staticmethod
    def hex_to_qcolor(hex_color):
        """Convert hex color string to QColor"""
        return QColor(hex_color)


def apply_application_style():
    """Apply the application-wide style"""
    # Set fusion style for a modern look across platforms
    QApplication.setStyle("Fusion")

    # Create a custom palette for dark theme
    palette = QPalette()

    # Set palette colors for dark theme
    palette.setColor(QPalette.Window, QColor(Colors.BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(Colors.TEXT))
    palette.setColor(QPalette.Base, QColor(Colors.WIDGET_BACKGROUND))
    palette.setColor(QPalette.AlternateBase, QColor(Colors.BLACK))
    palette.setColor(QPalette.ToolTipBase, QColor(Colors.BLACK))
    palette.setColor(QPalette.ToolTipText, QColor(Colors.LIGHT_BEIGE))
    palette.setColor(QPalette.Text, QColor(Colors.TEXT))
    palette.setColor(QPalette.Button, QColor(Colors.BUTTON))
    palette.setColor(QPalette.ButtonText, QColor(Colors.BUTTON_TEXT))
    palette.setColor(QPalette.BrightText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(Colors.HIGHLIGHT))
    palette.setColor(QPalette.HighlightedText, QColor(Colors.BLACK))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(Colors.DISABLED))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(Colors.DISABLED))

    # Apply the palette to the application instance
    app = QApplication.instance()
    if app is not None:
        app.setPalette(palette)

    # Set application font
    font = QFont("Segoe UI", 9)
    if app is not None:
        app.setFont(font)

    # Set stylesheet for fine-tuning
    stylesheet = """
    QMainWindow {
        background-color: #000000;
    }
    
    QWidget {
        background-color: #000000;
        color: #E6E5E0;
    }
    
    QMenuBar {
        background-color: #000000;
        color: #E6E5E0;
        border-bottom: 1px solid #333333;
    }
    
    QMenuBar::item:selected {
        background-color: #DBBB5F;
        color: #000000;
    }
    
    QMenu {
        background-color: #1A1A1A;
        color: #E6E5E0;
        border: 1px solid #333333;
    }
    
    QMenu::item:selected {
        background-color: #DBBB5F;
        color: #000000;
    }
    
    QPushButton {
        background-color: #DBBB5F;
        color: #000000;
        border: none;
        padding: 6px 12px;
        border-radius: 3px;
    }
    
    QPushButton:hover {
        background-color: #DFB011;
    }
    
    QPushButton:pressed {
        background-color: #D0C09E;
    }
    
    QPushButton:disabled {
        background-color: #666666;
        color: #CCCCCC;
    }
    
    QToolBar {
        background-color: #1A1A1A;
        border-bottom: 1px solid #333333;
        spacing: 3px;
    }
    
    QToolButton {
        background-color: transparent;
        color: #E6E5E0;
        border-radius: 3px;
        padding: 3px;
    }
    
    QToolButton:hover {
        background-color: #DBBB5F;
        color: #000000;
    }
    
    QToolButton:pressed {
        background-color: #D0C09E;
    }
    
    QStatusBar {
        background-color: #1A1A1A;
        color: #E6E5E0;
        border-top: 1px solid #333333;
    }
    
    QGroupBox {
        border: 1px solid #333333;
        border-radius: 3px;
        margin-top: 0.5em;
        padding-top: 0.5em;
        color: #E6E5E0;
    }
    
    QGroupBox::title {
        background-color: #000000;
        padding: 0 3px;
        color: #E6E5E0;
    }
    
    QLabel {
        color: #E6E5E0;
    }
    
    QCheckBox {
        color: #E6E5E0;
    }
    
    QCheckBox::indicator:checked {
        background-color: #DBBB5F;
    }
    
    QSlider::handle:horizontal {
        background: #DBBB5F;
        border: 1px solid #D0C09E;
        width: 10px;
        border-radius: 5px;
    }
    
    QSlider::groove:horizontal {
        background: #333333;
        height: 6px;
        border-radius: 3px;
    }
    
    QLineEdit, QComboBox {
        background-color: #1A1A1A;
        color: #E6E5E0;
        border: 1px solid #333333;
        border-radius: 3px;
        padding: 2px 4px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #1A1A1A;
        color: #E6E5E0;
        selection-background-color: #DBBB5F;
        selection-color: #000000;
    }
    
    QDockWidget {
        titlebar-close-icon: url(close.png);
        titlebar-normal-icon: url(undock.png);
        color: #E6E5E0;
    }
    
    QDockWidget::title {
        background-color: #333333;
        padding-left: 5px;
        padding-top: 2px;
        color: #E6E5E0;
    }
    
    QTabWidget::pane {
        border: 1px solid #333333;
        border-radius: 3px;
    }
    
    QTabBar::tab {
        background-color: #1A1A1A;
        color: #E6E5E0;
        border: 1px solid #333333;
        border-bottom: none;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
        padding: 5px 10px;
    }
    
    QTabBar::tab:selected {
        background-color: #DBBB5F;
        color: #000000;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #333333;
    }
    
    QSplitter::handle {
        background-color: #333333;
    }
    
    QTableView {
        background-color: #1A1A1A;
        color: #E6E5E0;
        gridline-color: #333333;
        selection-background-color: #DBBB5F;
        selection-color: #000000;
    }
    
    QHeaderView::section {
        background-color: #333333;
        color: #E6E5E0;
        padding: 4px;
        border: 1px solid #1A1A1A;
    }
    
    QProgressBar {
        border: 1px solid #333333;
        border-radius: 3px;
        background-color: #1A1A1A;
        color: #E6E5E0;
        text-align: center;
    }
    
    QProgressBar::chunk {
        background-color: #DBBB5F;
        width: 10px;
        margin: 0.5px;
    }
    """

    # Apply the stylesheet to the application instance
    if app is not None:
        app.setStyleSheet(stylesheet)