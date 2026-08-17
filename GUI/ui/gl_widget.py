#!/usr/bin/env python3
# OpenGL widget for rendering PLY models - Fixed version

import numpy as np
import math
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat, QImage
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from plyfile import PlyData


class GLWidget(QOpenGLWidget):
    """OpenGL widget for rendering PLY models with GPU acceleration"""

    file_loaded = pyqtSignal(bool)  # Signal to indicate when a file is loaded
    view_changed = pyqtSignal(float, float, float)  # Signal for rotation_x, rotation_y, zoom

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.vertices = None
        self.faces = None
        self.colors = None
        self.normals = None
        self.has_color = False
        self.draw_mode = GL_TRIANGLES
        self.show_wireframe = False
        self.show_points = False
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -3.0
        self.last_pos = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.bbox_min = None
        self.bbox_max = None
        self.current_file = None
        self.update_view_externally = False  # Flag to prevent signal loops

        # VBO objects for GPU acceleration
        self.vertex_vbo = None
        self.color_vbo = None
        self.normal_vbo = None
        self.index_vbo = None
        self.use_vbo = True  # Enable VBO by default

        # OpenGL context state
        self.gl_initialized = False

    def initializeGL(self):
        """Initialize OpenGL settings"""
        try:
            glClearColor(0.2, 0.2, 0.2, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Additional settings for better performance
            glEnable(GL_CULL_FACE)  # Enable face culling
            glCullFace(GL_BACK)  # Cull back faces

            # Check if we're in compatibility profile
            try:
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                self.using_fixed_pipeline = True
            except:
                print("Warning: Fixed function pipeline not available. Using fallback mode.")
                self.using_fixed_pipeline = False

            # Print OpenGL info
            print(f"OpenGL Vendor: {glGetString(GL_VENDOR).decode('utf-8')}")
            print(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode('utf-8')}")
            print(f"OpenGL Version: {glGetString(GL_VERSION).decode('utf-8')}")
            print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")
            print(f"Using fixed pipeline: {self.using_fixed_pipeline}")

            self.gl_initialized = True

        except Exception as e:
            print(f"Error initializing OpenGL: {e}")
            self.gl_initialized = False

    def resizeGL(self, width, height):
        """Handle OpenGL resize with proper error checking"""
        if not self.gl_initialized:
            return

        try:
            # Ensure we have a valid OpenGL context
            if not self.context() or not self.context().isValid():
                return

            glViewport(0, 0, width, height)

            if self.using_fixed_pipeline:
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                aspect = width / float(height) if height > 0 else 1.0
                gluPerspective(45.0, aspect, 0.1, 100.0)
                glMatrixMode(GL_MODELVIEW)
            else:
                # For core profile, we'd use shader-based approach
                # but for now just track the values we'd need for our calculations
                self.aspect_ratio = width / float(height) if height > 0 else 1.0

        except Exception as e:
            print(f"Error in resizeGL: {e}")

    def setup_vbos(self):
        """Create Vertex Buffer Objects for GPU-accelerated rendering"""
        if not self.gl_initialized:
            return False

        try:
            if self.vertices is not None and len(self.vertices) > 0:
                # Create and bind VBOs
                if self.vertex_vbo is None:
                    self.vertex_vbo = vbo.VBO(np.array(self.vertices, dtype=np.float32))

                if self.has_color and self.colors is not None and self.color_vbo is None:
                    self.color_vbo = vbo.VBO(np.array(self.colors, dtype=np.float32) / 255.0)

                if self.normals is not None and self.normal_vbo is None:
                    self.normal_vbo = vbo.VBO(np.array(self.normals, dtype=np.float32))

                if self.faces is not None and len(self.faces) > 0 and self.index_vbo is None:
                    # Flatten face indices for indexing
                    indices = np.array([idx for face in self.faces for idx in face], dtype=np.uint32)
                    self.index_vbo = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)

                return True
            return False
        except Exception as e:
            print(f"Error setting up VBOs: {e}")
            self.use_vbo = False
            return False

    def render_with_vbo(self):
        """Render the model using VBOs for GPU acceleration"""
        if not self.gl_initialized:
            return

        try:
            # Bind vertex data
            self.vertex_vbo.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)

            # Bind color data if available
            if self.has_color and self.color_vbo is not None:
                self.color_vbo.bind()
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_FLOAT, 0, None)

            # Bind normal data if available
            if self.normals is not None and self.normal_vbo is not None:
                self.normal_vbo.bind()
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, None)

            # Set polygon mode based on wireframe setting
            if self.show_wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Render points if requested
            if self.show_points:
                glPointSize(3.0)
                glDrawArrays(GL_POINTS, 0, len(self.vertices))

            # Draw faces using indexed rendering if available
            if self.faces is not None and len(self.faces) > 0 and self.index_vbo is not None:
                self.index_vbo.bind()
                # Calculate the total number of indices
                total_indices = sum(len(face) for face in self.faces)
                glDrawElements(self.draw_mode, total_indices, GL_UNSIGNED_INT, None)
                self.index_vbo.unbind()

            # Reset polygon mode
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Unbind VBOs and disable states
            if self.normals is not None and self.normal_vbo is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
                self.normal_vbo.unbind()

            if self.has_color and self.color_vbo is not None:
                glDisableClientState(GL_COLOR_ARRAY)
                self.color_vbo.unbind()

            glDisableClientState(GL_VERTEX_ARRAY)
            self.vertex_vbo.unbind()

        except Exception as e:
            print(f"Error rendering with VBO: {e}")
            self.use_vbo = False
            self.render_with_immediate_mode()

    def render_with_immediate_mode(self):
        """Fallback to immediate mode rendering if VBO fails"""
        if not self.gl_initialized or self.vertices is None or len(self.vertices) == 0:
            return

        try:
            # Render points
            if self.show_points:
                glPointSize(3.0)
                glBegin(GL_POINTS)
                for i, vertex in enumerate(self.vertices):
                    if self.has_color and self.colors is not None:
                        glColor3f(self.colors[i][0] / 255, self.colors[i][1] / 255, self.colors[i][2] / 255)
                    else:
                        glColor3f(1.0, 1.0, 1.0)
                    glVertex3f(vertex[0], vertex[1], vertex[2])
                glEnd()

            # Render faces or wireframe
            if self.faces is not None and len(self.faces) > 0:
                if self.show_wireframe:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    glColor3f(1.0, 1.0, 1.0)
                else:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                glBegin(self.draw_mode)
                for face in self.faces:
                    # Calculate normal for the face if not provided
                    if self.normals is None:
                        v0 = self.vertices[face[0]]
                        v1 = self.vertices[face[1]]
                        v2 = self.vertices[face[2]]

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
                        if self.normals is not None:
                            glNormal3f(self.normals[vertex_idx][0], self.normals[vertex_idx][1],
                                       self.normals[vertex_idx][2])

                        if self.has_color and self.colors is not None:
                            glColor3f(self.colors[vertex_idx][0] / 255, self.colors[vertex_idx][1] / 255,
                                      self.colors[vertex_idx][2] / 255)
                        else:
                            glColor3f(0.8, 0.8, 0.8)

                        glVertex3f(self.vertices[vertex_idx][0], self.vertices[vertex_idx][1],
                                   self.vertices[vertex_idx][2])
                glEnd()

                # Reset polygon mode
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        except Exception as e:
            print(f"Error in immediate mode rendering: {e}")

    def paintGL(self):
        """Main OpenGL paint function with error handling"""
        if not self.gl_initialized:
            return

        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if not self.using_fixed_pipeline:
                return

            glLoadIdentity()

            # Position the camera
            glTranslatef(0.0, 0.0, self.zoom)
            glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
            glRotatef(self.rotation_y, 0.0, 1.0, 0.0)

            # Draw coordinate axes (red = x, green = y, blue = z)
            glBegin(GL_LINES)
            # X axis (red)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(1.0, 0.0, 0.0)
            # Y axis (green)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, 1.0, 0.0)
            # Z axis (blue)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, 1.0)
            glEnd()

            # Render the model if data is available
            if self.vertices is not None and len(self.vertices) > 0:
                # Center the model
                if self.bbox_min is not None and self.bbox_max is not None:
                    center = [(self.bbox_min[i] + self.bbox_max[i]) / 2 for i in range(3)]
                    glTranslatef(-center[0], -center[1], -center[2])

                # Choose rendering method based on GPU acceleration flag
                if self.use_vbo and self.setup_vbos():
                    self.render_with_vbo()
                else:
                    self.render_with_immediate_mode()

        except Exception as e:
            print(f"Error in paintGL: {e}")

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()

            if event.buttons() & Qt.LeftButton:
                self.rotation_y += dx
                self.rotation_x += dy
                self.update()
                # Emit signal for view change
                if not self.update_view_externally:
                    self.view_changed.emit(self.rotation_x, self.rotation_y, self.zoom)
            elif event.buttons() & Qt.RightButton:
                self.zoom += dy / 100.0
                self.update()
                # Emit signal for view change
                if not self.update_view_externally:
                    self.view_changed.emit(self.rotation_x, self.rotation_y, self.zoom)

            self.last_pos = event.pos()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom += delta * 0.2
        self.update()
        # Emit signal for view change
        if not self.update_view_externally:
            self.view_changed.emit(self.rotation_x, self.rotation_y, self.zoom)

    def set_view(self, rotation_x, rotation_y, zoom):
        """Set the view parameters externally"""
        self.update_view_externally = True
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.zoom = zoom
        self.update()
        self.update_view_externally = False

    def load_ply(self, filename):
        """Load a PLY file and prepare it for rendering"""
        try:
            self.current_file = filename
            plydata = PlyData.read(filename)

            # Extract vertices
            vertex_data = plydata['vertex']
            self.vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

            # Check for colors
            self.has_color = False
            if 'red' in vertex_data and 'green' in vertex_data and 'blue' in vertex_data:
                self.colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
                self.has_color = True

            # Check for normals
            if 'nx' in vertex_data and 'ny' in vertex_data and 'nz' in vertex_data:
                self.normals = np.vstack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']]).T
            else:
                self.normals = None

            # Extract faces if available
            self.faces = []
            if 'face' in plydata:
                face_data = plydata['face']
                if 'vertex_indices' in face_data:
                    self.faces = [face for face in face_data['vertex_indices']]
                elif 'vertex_index' in face_data:  # Some PLY files use this format
                    self.faces = [face for face in face_data['vertex_index']]

            # Compute bounding box
            self.bbox_min = [min(self.vertices[:, i]) for i in range(3)]
            self.bbox_max = [max(self.vertices[:, i]) for i in range(3)]

            # Normalize model to fit into view
            self.center_and_scale()

            # Clean up any previous VBOs
            self.vertex_vbo = None
            self.color_vbo = None
            self.normal_vbo = None
            self.index_vbo = None

            # Emit signal that file is loaded
            self.file_loaded.emit(True)

            self.update()
            return True
        except Exception as e:
            print(f"Error loading PLY file: {e}")
            return False

    def center_and_scale(self):
        """Center and scale the model to fit in view"""
        # Calculate model dimensions
        dimensions = [self.bbox_max[i] - self.bbox_min[i] for i in range(3)]
        max_dim = max(dimensions)

        if max_dim > 0:
            # Scale to fit in a 2x2x2 box
            scale_factor = 2.0 / max_dim

            # Scale vertices
            self.vertices = self.vertices * scale_factor

            # Update bounding box
            self.bbox_min = [min(self.vertices[:, i]) for i in range(3)]
            self.bbox_max = [max(self.vertices[:, i]) for i in range(3)]

    def render_to_image(self, width=512, height=512):
        """Render the current view to an image with proper context handling"""
        if not self.gl_initialized:
            print("OpenGL not initialized, cannot render to image")
            return QImage(width, height, QImage.Format_RGB32)

        try:
            # Make sure we have a current OpenGL context
            self.makeCurrent()

            # Check if context is valid
            if not self.context() or not self.context().isValid():
                print("Invalid OpenGL context")
                return QImage(width, height, QImage.Format_RGB32)

            # Create framebuffer object
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setAttachment(QOpenGLFramebufferObject.Depth)
            fbo = QOpenGLFramebufferObject(width, height, fmt)

            if not fbo.isValid():
                print("Failed to create framebuffer object")
                return QImage(width, height, QImage.Format_RGB32)

            # Bind framebuffer and render
            fbo.bind()

            # Save current viewport
            current_viewport = glGetIntegerv(GL_VIEWPORT)

            # Set viewport for rendering
            self.resizeGL(width, height)
            self.paintGL()

            # Get the image
            image = fbo.toImage()

            # Restore original viewport
            if current_viewport is not None:
                glViewport(*current_viewport)

            # Release framebuffer
            fbo.release()

            # Restore original context
            self.doneCurrent()

            return image

        except Exception as e:
            print(f"Error rendering to image: {e}")
            # Return a blank image in case of error
            return QImage(width, height, QImage.Format_RGB32)

    def get_model_info(self):
        """Return basic information about the loaded model"""
        if self.vertices is None:
            return None

        return {
            'filename': self.current_file,
            'vertices': len(self.vertices),
            'faces': len(self.faces) if self.faces else 0,
            'has_color': self.has_color,
            'has_normals': self.normals is not None
        }