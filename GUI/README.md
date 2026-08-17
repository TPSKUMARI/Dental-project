# PLY Viewer and Comparator

A Python application for viewing and comparing 3D PLY files with GPU acceleration.

## Features

- Interactive 3D viewing of PLY files
- GPU-accelerated rendering using OpenGL VBOs
- Side-by-side comparison of two PLY models
- Multiple comparison methods:
  - Exact match (pixel-by-pixel)
  - Intensity-based comparison
  - Color channel analysis
  - Perceptual difference (weighted RGB)
- Adjustable thresholds for each comparison method
- Detailed statistics and visualizations

## Requirements

- Python 3.6+
- numpy
- PyQt5
- PyOpenGL
- PyOpenGL_accelerate
- plyfile
- matplotlib
- scipy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/dental-project.git
   cd ply-viewer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python main.py
```

### Loading Models

1. Use "File > Open Model 1" (or Ctrl+1) to load the first PLY file
2. Use "File > Open Model 2" (or Ctrl+2) to load the second PLY file

### Navigation

- Rotate: Left mouse button + drag
- Zoom: Mouse wheel or right mouse button + drag
- Check "Sync Camera Views" to keep both models in the same orientation

### Display Options

- Toggle wireframe mode
- Show/hide points
- Enable/disable GPU acceleration (VBO)

### Comparison

1. Select a comparison method from the dropdown
2. Adjust the threshold sliders as needed
3. Click "Compare Models" (or use Ctrl+C)
4. A new window will show both models and their pixel-by-pixel differences

### Analysis

1. 
2. 
3. 
4. 
5. 

## Project Structure

```
ply-viewer/
├── main.py                          # Main application entry point with dark theme integration
├── config.py                        # OpenGL configuration
├── theme_style.py                   # NEW: Dark theme stylesheet definitions
├── __init__.py                      # Package initialization
├── requirements.txt                 # Dependencies (includes scipy, PyQt5, etc.)
├── README.md                        # Documentation
├── ui/                              # User interface components
│   ├── __init__.py                  # Package initialization for UI
│   ├── main_window.py               # UPDATED: Main window with enhanced dark theme styling
│   ├── gl_widget.py                 # OpenGL widget for rendering PLY models
│   ├── comparison_window.py         # Static comparison window
│   ├── live_comparison_window.py    # Window for real-time comparison
│   ├── comparison_controls.py       # UPDATED: Enhanced tartar detection controls with styling
│   ├── point_cloud_viewer.py        # Custom OpenGL widget for point cloud visualization
│   └── tartar_visualization_dialog.py # Dialog for visualizing dental tartar analysis
└── utils/                           # Utility functions
    ├── __init__.py                  # Package initialization for utils
    ├── comparison_methods.py        # UPDATED: Simplified to only include tartar detection
    └── ply_utils.py                 # PLY file handling and tartar detection functions
```

## License

"XXXXX"
