"""
OpenGL widget for rendering PLY models with GPU acceleration
"""

import numpy as np
import math
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from utils.gl_utils import print_gl_info
from ui.style import Colors


class GLWidget(QOpenGLWidget):
    """OpenGL widget for rendering PLY models with GPU acceleration"""

    # Signals
    modelLoaded = pyqtSignal()
    viewChanged = pyqtSignal()

    # Drawing modes
    DRAW_POINTS = 0
    DRAW_WIREFRAME = 1
    DRAW_SOLID = 2
    DRAW_TEXTURED = 3

    # Background color - Black
    BACKGROUND_COLOR = (0.0, 0.0, 0.0)  # #000000

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        # Model
        self.model = None

        # Rendering options
        self.draw_mode = self.DRAW_SOLID
        self.show_wireframe = False
        self.show_points = False
        self.show_axes = True
        self.show_bounding_box = False
        self.use_vbo = True  # Enable VBO by default

        # Point and line size properties
        self.point_size = 3.0
        self.line_width = 1.0

        # Camera/view parameters
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -3.0
        self.last_pos = None

        # Colors - Updated for black background
        self.background_color = self.BACKGROUND_COLOR
        self.wireframe_color = Colors.hex_to_rgb(Colors.LIGHT_BEIGE)  # Light beige wireframes
        self.point_color = Colors.hex_to_rgb(Colors.LIGHT_BEIGE)  # Light beige points
        self.default_color = Colors.hex_to_rgb(Colors.LIGHT_BEIGE)  # Light beige default
        self.axes_colors = [
            Colors.hex_to_rgb(Colors.DARK_GOLD),   # X axis (DFB011)
            Colors.hex_to_rgb(Colors.TAN),         # Y axis (D0C09E)
            Colors.hex_to_rgb(Colors.LIGHT_BEIGE)  # Z axis (E6E5E0)
        ]

        self.setFocusPolicy(Qt.StrongFocus)

    # In gl_widget.py, update the set_model method:
    def set_model(self, model):
        """Set the 3D model to render"""
        # Clean up previous VBOs more thoroughly
        if self.model is not None:
            self.cleanup_vbos()
            # Force garbage collection
            import gc
            gc.collect()

        self.model = model

        # Only setup VBOs if model is loaded
        if self.model and self.model.is_loaded():
            # Add a small delay to ensure GL context is ready
            QTimer.singleShot(10, self.setup_vbos)
            self.modelLoaded.emit()

        self.update()

    def clear_model(self):
        """Clear the current model and reset VBOs"""
        try:
            print("Clearing GL widget model...")

            # Clear VBOs if they exist
            if self.model:  # Add this check
                if hasattr(self.model, 'vertex_vbo') and self.model.vertex_vbo:
                    try:
                        self.model.vertex_vbo.delete()
                    except:
                        pass
                if hasattr(self.model, 'color_vbo') and self.model.color_vbo:
                    try:
                        self.model.color_vbo.delete()
                    except:
                        pass
                if hasattr(self.model, 'normal_vbo') and self.model.normal_vbo:
                    try:
                        self.model.normal_vbo.delete()
                    except:
                        pass
                if hasattr(self.model, 'index_vbo') and self.model.index_vbo:
                    try:
                        self.model.index_vbo.delete()
                    except:
                        pass

            # Clear the model
            self.model = None

            print("GL widget model cleared successfully")

        except Exception as e:
            print(f"Error clearing model: {e}")

    def initializeGL(self):
        """Initialize OpenGL settings"""
        # Set background color to black
        glClearColor(*self.background_color, 1.0)

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable face culling for better performance
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Enable point size and line width
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Set initial point size and line width
        glPointSize(self.point_size)
        glLineWidth(self.line_width)

        # Print OpenGL info
        print_gl_info()

    def resizeGL(self, width, height):
        """Handle window resize events"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height)
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Apply current point size and line width
        glPointSize(self.point_size)
        glLineWidth(self.line_width)

        # Position the camera
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)

        # Draw coordinate axes if enabled
        if self.show_axes:
            self.draw_axes()

        # Render the model if available
        if self.model and self.model.is_loaded():
            # Center the model
            if self.model.bbox_min is not None and self.model.bbox_max is not None:
                center = [(self.model.bbox_min[i] + self.model.bbox_max[i]) / 2 for i in range(3)]
                glTranslatef(-center[0], -center[1], -center[2])

                # Draw bounding box if enabled
                if self.show_bounding_box:
                    self.draw_bounding_box()

            # Choose rendering method based on GPU acceleration flag
            if self.use_vbo and self.setup_vbos():
                self.render_with_vbo()
            else:
                self.render_with_immediate_mode()

    def draw_axes(self):
        """Draw coordinate axes"""
        axis_length = 1.0

        # Save current line width
        current_line_width = self.line_width

        # Use a fixed line width for axes to ensure visibility
        glLineWidth(max(2.0, current_line_width))

        glBegin(GL_LINES)
        # X axis
        glColor3f(*self.axes_colors[0])
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)

        # Y axis
        glColor3f(*self.axes_colors[1])
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)

        # Z axis
        glColor3f(*self.axes_colors[2])
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)
        glEnd()

        # Restore original line width
        glLineWidth(current_line_width)

    def draw_bounding_box(self):
        """Draw bounding box around the model"""
        if self.model.bbox_min is None or self.model.bbox_max is None:
            return

        min_x, min_y, min_z = self.model.bbox_min
        max_x, max_y, max_z = self.model.bbox_max

        # Set color for bounding box
        glColor3f(*self.wireframe_color)

        # Draw lines connecting bounding box vertices
        glBegin(GL_LINES)
        # Bottom face
        glVertex3f(min_x, min_y, min_z)
        glVertex3f(max_x, min_y, min_z)

        glVertex3f(max_x, min_y, min_z)
        glVertex3f(max_x, min_y, max_z)

        glVertex3f(max_x, min_y, max_z)
        glVertex3f(min_x, min_y, max_z)

        glVertex3f(min_x, min_y, max_z)
        glVertex3f(min_x, min_y, min_z)

        # Top face
        glVertex3f(min_x, max_y, min_z)
        glVertex3f(max_x, max_y, min_z)

        glVertex3f(max_x, max_y, min_z)
        glVertex3f(max_x, max_y, max_z)

        glVertex3f(max_x, max_y, max_z)
        glVertex3f(min_x, max_y, max_z)

        glVertex3f(min_x, max_y, max_z)
        glVertex3f(min_x, max_y, min_z)

        # Connecting edges
        glVertex3f(min_x, min_y, min_z)
        glVertex3f(min_x, max_y, min_z)

        glVertex3f(max_x, min_y, min_z)
        glVertex3f(max_x, max_y, min_z)

        glVertex3f(max_x, min_y, max_z)
        glVertex3f(max_x, max_y, max_z)

        glVertex3f(min_x, min_y, max_z)
        glVertex3f(min_x, max_y, max_z)
        glEnd()

    def setup_vbos(self):
        """Create Vertex Buffer Objects for GPU-accelerated rendering"""
        if not self.model or not self.model.is_loaded():
            return False

        try:
            # Make context current
            self.makeCurrent()

            # Clean up any existing VBOs first
            self.cleanup_vbos()

            vertices = self.model.vertices

            # Check if we actually need to create new VBOs
            if vertices is None or len(vertices) == 0:
                return False

            print(f"Setting up VBOs for {len(vertices)} vertices...")

            # Create VBOs only if data exists
            if vertices is not None and len(vertices) > 0:
                # Use STATIC_DRAW for better performance
                self.model.vertex_vbo = vbo.VBO(
                    np.array(vertices, dtype=np.float32).copy(),
                    usage=GL_STATIC_DRAW
                )

            if self.model.has_color and self.model.colors is not None:
                # Ensure colors are properly formatted
                colors = np.array(self.model.colors, dtype=np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                self.model.color_vbo = vbo.VBO(colors.copy(), usage=GL_STATIC_DRAW)

            # ... rest of VBO setup

            print("VBO setup completed successfully")
            return True

        except Exception as e:
            print(f"Error setting up VBOs: {e}")
            import traceback
            traceback.print_exc()
            self.use_vbo = False
            return False

    def cleanup_vbos(self):
        """Clean up existing VBOs before creating new ones"""
        if not self.model:
            return

        try:
            # Delete existing VBOs
            if hasattr(self.model, 'vertex_vbo') and self.model.vertex_vbo:
                self.model.vertex_vbo.delete()
                self.model.vertex_vbo = None

            if hasattr(self.model, 'color_vbo') and self.model.color_vbo:
                self.model.color_vbo.delete()
                self.model.color_vbo = None

            if hasattr(self.model, 'normal_vbo') and self.model.normal_vbo:
                self.model.normal_vbo.delete()
                self.model.normal_vbo = None

            if hasattr(self.model, 'index_vbo') and self.model.index_vbo:
                self.model.index_vbo.delete()
                self.model.index_vbo = None

        except Exception as e:
            print(f"Error cleaning up VBOs: {e}")

    def render_with_vbo(self):
        """Render the model using VBOs for GPU acceleration"""
        if not self.model or not self.model.is_loaded():
            return

        try:
            # Bind vertex data
            self.model.vertex_vbo.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)

            # Bind color data if available
            if self.model.has_color and self.model.color_vbo is not None:
                self.model.color_vbo.bind()
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_FLOAT, 0, None)
            else:
                # Use default color
                glColor3f(*self.default_color)

            # Bind normal data if available
            if self.model.normals is not None and self.model.normal_vbo is not None:
                self.model.normal_vbo.bind()
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, None)

            # Set polygon mode based on wireframe setting
            if self.show_wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glColor3f(*self.wireframe_color)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Render points if requested
            if self.show_points:
                # Explicitly set point size
                glPointSize(self.point_size)
                glColor3f(*self.point_color)
                glDrawArrays(GL_POINTS, 0, len(self.model.vertices))

            # Draw faces using indexed rendering if available
            if (self.model.faces is not None and len(self.model.faces) > 0 and
                    self.model.index_vbo is not None):
                self.model.index_vbo.bind()

                # Calculate the total number of indices
                total_indices = sum(len(face) for face in self.model.faces)

                # Draw the model
                glDrawElements(GL_TRIANGLES, total_indices, GL_UNSIGNED_INT, None)

                self.model.index_vbo.unbind()

            # Reset polygon mode
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Unbind VBOs and disable states
            if self.model.normals is not None and self.model.normal_vbo is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
                self.model.normal_vbo.unbind()

            if self.model.has_color and self.model.color_vbo is not None:
                glDisableClientState(GL_COLOR_ARRAY)
                self.model.color_vbo.unbind()

            glDisableClientState(GL_VERTEX_ARRAY)
            self.model.vertex_vbo.unbind()

        except Exception as e:
            print(f"Error rendering with VBO: {e}")
            self.use_vbo = False
            self.render_with_immediate_mode()

    def render_with_immediate_mode(self):
        """Fallback to immediate mode rendering if VBO fails"""
        if not self.model or not self.model.is_loaded():
            return

        vertices = self.model.vertices
        faces = self.model.faces
        colors = self.model.colors
        normals = self.model.normals
        has_color = self.model.has_color

        # Render points
        if self.show_points:
            # Explicitly set point size
            glPointSize(self.point_size)
            glColor3f(*self.point_color)
            glBegin(GL_POINTS)
            for i, vertex in enumerate(vertices):
                if has_color and colors is not None:
                    glColor3f(colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255)
                glVertex3f(vertex[0], vertex[1], vertex[2])
            glEnd()

        # Render faces or wireframe
        if faces is not None and len(faces) > 0:
            if self.show_wireframe:
                # Explicitly set line width
                glLineWidth(self.line_width)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glColor3f(*self.wireframe_color)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                if not has_color:
                    glColor3f(*self.default_color)

            glBegin(GL_TRIANGLES)
            for face in faces:
                # Calculate normal for the face if not provided
                if normals is None:
                    v0 = vertices[face[0]]
                    v1 = vertices[face[1]]
                    v2 = vertices[face[2]]

                    # Simple normal calculation
                    edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]]
                    edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]]
                    normal = [
                        edge1[1] * edge2[2] - edge1[2] * edge2[1],
                        edge1[2] * edge2[0] - edge1[0] * edge2[2],
                        edge1[0] * edge2[1] - edge1[1] * edge2[0]
                    ]

                    # Normalize
                    length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
                    if length > 0:
                        normal = [n / length for n in normal]

                    glNormal3f(normal[0], normal[1], normal[2])

                for vertex_idx in face:
                    if normals is not None:
                        glNormal3f(normals[vertex_idx][0], normals[vertex_idx][1],
                                   normals[vertex_idx][2])

                    if has_color and colors is not None:
                        glColor3f(colors[vertex_idx][0] / 255, colors[vertex_idx][1] / 255,
                                  colors[vertex_idx][2] / 255)

                    glVertex3f(vertices[vertex_idx][0], vertices[vertex_idx][1],
                               vertices[vertex_idx][2])
            glEnd()

            # Reset polygon mode
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def set_point_size(self, size):
        """Set the point size for rendering"""
        self.point_size = float(size)
        self.update()

    def set_line_width(self, width):
        """Set the line width for wireframe rendering"""
        self.line_width = float(width)
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for rotation and zoom"""
        if self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()

            if event.buttons() & Qt.LeftButton:
                self.rotation_y += dx
                self.rotation_x += dy
                self.update()
                self.viewChanged.emit()
            elif event.buttons() & Qt.RightButton:
                self.zoom += dy / 100.0
                self.update()
                self.viewChanged.emit()

            self.last_pos = event.pos()

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        delta = event.angleDelta().y() / 120
        self.zoom += delta * 0.2
        self.update()
        self.viewChanged.emit()

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        key = event.key()

        if key == Qt.Key_R:
            # Reset view
            self.rotation_x = 0
            self.rotation_y = 0
            self.zoom = -3.0
            self.update()
            self.viewChanged.emit()
        elif key == Qt.Key_W:
            # Toggle wireframe
            self.show_wireframe = not self.show_wireframe
            self.update()
        elif key == Qt.Key_P:
            # Toggle points
            self.show_points = not self.show_points
            self.update()
        elif key == Qt.Key_B:
            # Toggle bounding box
            self.show_bounding_box = not self.show_bounding_box
            self.update()
        elif key == Qt.Key_A:
            # Toggle axes
            self.show_axes = not self.show_axes
            self.update()
        else:
            super(GLWidget, self).keyPressEvent(event)

    def set_background_color(self, color):
        """Set the background color of the OpenGL widget"""
        self.background_color = color
        # We'll only set the OpenGL color in the initializeGL method or when actually rendering
        # to avoid calling OpenGL functions before the context is created
        self.update()
        
    def toggle_wireframe(self, enabled):
        """Toggle wireframe rendering mode"""
        self.show_wireframe = enabled
        self.update()
        
    def toggle_points(self, enabled):
        """Toggle point rendering mode"""
        self.show_points = enabled
        self.update()
        
    def toggle_axes(self, enabled):
        """Toggle coordinate axes display"""
        self.show_axes = enabled
        self.update()
        
    def toggle_bounding_box(self, enabled):
        """Toggle bounding box display"""
        self.show_bounding_box = enabled
        self.update()
        
    def toggle_vbo(self, enabled):
        """Toggle GPU acceleration (VBO)"""
        self.use_vbo = enabled
        self.update()

    def has_model(self):
        """Check if a model is currently loaded"""
        return self.model is not None and self.model.is_loaded()

    def get_model_info(self):
        """Get information about the currently loaded model"""
        if not self.has_model():
            return None

        return {
            'vertices': len(self.model.vertices) if self.model.vertices is not None else 0,
            'faces': len(self.model.faces) if self.model.faces is not None else 0,
            'has_colors': self.model.has_color,
            'has_normals': self.model.normals is not None,
            'bbox_min': self.model.bbox_min,
            'bbox_max': self.model.bbox_max
        }

    def force_redraw(self):
        """Force an immediate redraw of the OpenGL widget"""
        self.update()
        self.repaint()


    def optimize_for_large_models(self):
        """Apply optimizations for handling large models efficiently"""
        try:
            # Enable point sprite extension for faster point rendering
            if self.show_points:
                glEnable(GL_POINT_SPRITE)
                glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

            # Use simpler lighting model for performance
            glShadeModel(GL_FLAT)

            # Disable expensive operations
            glDisable(GL_LIGHTING)
            glDisable(GL_DITHER)

            # Set polygon mode to fill for better performance
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Use fastest render quality
            glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST)
            glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST)
            glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST)

            return True
        except Exception as e:
            print(f"Error applying large model optimizations: {e}")
            return False

    def use_gpu_based_filtering(self, filter_func=None):
        """
        Enable GPU-based filtering for large models

        Args:
            filter_func: Optional function to determine visibility of vertices
        """
        # This is a placeholder for future implementation
        # In a real implementation, this would use OpenGL shaders to
        # perform filtering directly on the GPU
        pass

    def setup_hardware_buffers(self):
        """Set up advanced hardware buffers for better GPU performance"""
        try:
            if not self.model or not self.model.vertices:
                return False

            # Check if model has data
            vertices = self.model.vertices
            if len(vertices) == 0:
                return False

            # Create and bind VBOs with optimized flags
            if self.model.vertex_vbo is None:
                # Use GL_STATIC_DRAW for vertex data that doesn't change often
                self.model.vertex_vbo = vbo.VBO(
                    np.array(vertices, dtype=np.float32),
                    usage=GL_STATIC_DRAW
                )

            if self.model.color_vbo is None and self.model.has_color and self.model.colors is not None:
                # Normalize colors to 0-1 range for OpenGL
                self.model.color_vbo = vbo.VBO(
                    np.array(self.model.colors, dtype=np.float32) / 255.0,
                    usage=GL_STATIC_DRAW
                )

            if self.model.normal_vbo is None and self.model.normals is not None:
                self.model.normal_vbo = vbo.VBO(
                    np.array(self.model.normals, dtype=np.float32),
                    usage=GL_STATIC_DRAW
                )

            if (self.model.index_vbo is None and self.model.faces is not None and
                    len(self.model.faces) > 0):
                # Flatten face indices for indexing
                indices = np.array([idx for face in self.model.faces for idx in face], dtype=np.uint32)
                self.model.index_vbo = vbo.VBO(
                    indices,
                    target=GL_ELEMENT_ARRAY_BUFFER,
                    usage=GL_STATIC_DRAW
                )

            return True
        except Exception as e:
            print(f"Error setting up hardware buffers: {e}")
            return False