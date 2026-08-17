#!/usr/bin/env python3

# Custom OpenGL widget for fast point cloud visualization



import numpy as np

from PyQt5.QtWidgets import QOpenGLWidget

from PyQt5.QtCore import Qt, QPoint, QSize

from PyQt5.QtGui import QMatrix4x4, QVector3D, QOpenGLShaderProgram, QOpenGLShader

import OpenGL.GL as gl

import OpenGL.GLU as glu  # Add this import for gluPerspective





class PointCloudViewer(QOpenGLWidget):

    """

    Custom OpenGL widget for efficient point cloud visualization.

    Optimized for displaying large dental models with tartar highlighting.

    Uses a more compatible approach for different OpenGL implementations.

    """



    def __init__(self, parent=None):

        super(PointCloudViewer, self).__init__(parent)



        # Data attributes

        self.vertices = None

        self.colors = None

        self.highlight_mask = None



        # OpenGL attributes

        self.shader_program = None

        self.vbo_vertices = None

        self.vbo_colors = None

        self.vbo_highlight = None



        # Camera attributes

        self.rotation_x = 0.0

        self.rotation_y = 0.0

        self.scale = 1.0

        self.translation = [0.0, 0.0, -5.0]



        # Mouse interaction

        self.last_pos = QPoint()

        self.setFocusPolicy(Qt.StrongFocus)



        # Point rendering

        self.point_size = 2.0

        self.highlight_color = [1.0, 0.0, 0.0]  # Red for tartar

        self.background_color = [0.2, 0.2, 0.2]  # Dark gray



    def set_data(self, vertices, colors, highlight_mask=None):

        """Set the point cloud data for visualization"""

        self.vertices = vertices.astype(np.float32)



        # Normalize colors to 0-1 range if they're in 0-255 range

        if np.max(colors) > 1.0:

            self.colors = (colors / 255.0).astype(np.float32)

        else:

            self.colors = colors.astype(np.float32)



        # Set highlight mask (e.g., tartar areas)

        if highlight_mask is not None:

            self.highlight_mask = highlight_mask.astype(np.float32)

        else:

            self.highlight_mask = np.zeros(len(vertices), dtype=np.float32)



        # Center and scale the point cloud to fit the view

        self.center_and_scale()



        # Trigger a redraw

        self.update()



    def center_and_scale(self):

        """Center and scale the point cloud to fit in view"""

        if self.vertices is None or len(self.vertices) == 0:

            return



        # Calculate the center of the point cloud

        center = np.mean(self.vertices, axis=0)



        # Center the points around the origin

        self.vertices = self.vertices - center



        # Scale to fit in view

        max_extent = np.max(np.abs(self.vertices))

        if max_extent > 0:

            self.vertices = self.vertices / max_extent * 2.0



    def initializeGL(self):

        """Initialize OpenGL settings"""

        # Set clear color

        gl.glClearColor(*self.background_color, 1.0)



        # Enable depth testing

        gl.glEnable(gl.GL_DEPTH_TEST)



        # Enable point size

        gl.glEnable(gl.GL_POINT_SMOOTH)

        gl.glEnable(gl.GL_BLEND)

        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)



    def resizeGL(self, width, height):

        """Handle window resize event"""

        # Update viewport

        gl.glViewport(0, 0, width, height)



        # Setup projection matrix

        gl.glMatrixMode(gl.GL_PROJECTION)

        gl.glLoadIdentity()



        # Calculate aspect ratio

        aspect = width / float(height or 1)



        # Set perspective projection

        glu.gluPerspective(45.0, aspect, 0.1, 100.0)  # Use glu instead of gl



        # Switch back to modelview matrix

        gl.glMatrixMode(gl.GL_MODELVIEW)



    def paintGL(self):

        """Render the point cloud using immediate mode for compatibility"""

        # Clear the screen

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)



        # Check if there's data to render

        if self.vertices is None or len(self.vertices) == 0:

            return



        # Set up modelview matrix

        gl.glLoadIdentity()



        # Apply camera transformations

        gl.glTranslatef(*self.translation)

        gl.glRotatef(self.rotation_x, 1.0, 0.0, 0.0)

        gl.glRotatef(self.rotation_y, 0.0, 1.0, 0.0)

        gl.glScalef(self.scale, self.scale, self.scale)



        # Set point size

        gl.glPointSize(self.point_size)



        # Draw points

        gl.glBegin(gl.GL_POINTS)



        # Draw each point

        for i in range(len(self.vertices)):

            # Set color based on highlight mask

            if self.highlight_mask[i] > 0.5:

                gl.glColor3f(*self.highlight_color)

            else:

                gl.glColor3f(*self.colors[i])



            # Set vertex position

            gl.glVertex3f(*self.vertices[i])



        gl.glEnd()



    def mousePressEvent(self, event):

        """Handle mouse press event for model rotation"""

        self.last_pos = event.pos()



    def mouseMoveEvent(self, event):

        """Handle mouse move event for model rotation and panning"""

        dx = event.x() - self.last_pos.x()

        dy = event.y() - self.last_pos.y()



        # Rotate with left button

        if event.buttons() & Qt.LeftButton:

            self.rotation_y += dx * 0.5

            self.rotation_x += dy * 0.5

            self.update()



        # Pan with right button

        elif event.buttons() & Qt.RightButton:

            self.translation[0] += dx * 0.01

            self.translation[1] -= dy * 0.01

            self.update()



        self.last_pos = event.pos()



    def wheelEvent(self, event):

        """Handle mouse wheel event for zooming"""

        # Get number of degrees rotated in eighths of a degree

        delta = event.angleDelta().y()



        # Adjust scale factor (zoom)

        if delta > 0:

            self.scale *= 1.1

        else:

            self.scale *= 0.9



        self.update()



    def keyPressEvent(self, event):

        """Handle key press events for additional interactions"""

        # Reset view with 'R' key

        if event.key() == Qt.Key_R:

            self.reset_view()



        # Adjust point size with +/- keys

        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:

            self.point_size += 0.5

            self.update()



        elif event.key() == Qt.Key_Minus:

            self.point_size = max(0.5, self.point_size - 0.5)

            self.update()



    def reset_view(self):

        """Reset camera to default position"""

        self.rotation_x = 0.0

        self.rotation_y = 0.0

        self.scale = 1.0

        self.translation = [0.0, 0.0, -5.0]

        self.update()



    def sizeHint(self):

        """Suggested default size for the widget"""

        return QSize(640, 480)