"""
OpenGL utility functions for the PLY Viewer
"""

from PyQt5.QtGui import QSurfaceFormat


def configure_gl_format():
    """Configure OpenGL format for better GPU performance"""
    gl_format = QSurfaceFormat()
    # Use OpenGL 3.3 for better compatibility
    gl_format.setVersion(3, 3)
    # Use compatibility profile instead of core profile
    gl_format.setProfile(QSurfaceFormat.CompatibilityProfile)
    gl_format.setSamples(4)  # 4x MSAA
    gl_format.setDepthBufferSize(24)
    gl_format.setStencilBufferSize(8)
    gl_format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(gl_format)


def print_gl_info():
    """Print OpenGL driver and version information"""
    from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION, GL_SHADING_LANGUAGE_VERSION
    
    try:
        print(f"OpenGL Vendor: {glGetString(GL_VENDOR).decode('utf-8')}")
        print(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode('utf-8')}")
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode('utf-8')}")
        print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")
    except Exception as e:
        print(f"Error getting OpenGL info: {e}")
