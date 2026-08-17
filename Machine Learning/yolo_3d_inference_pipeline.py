#!/usr/bin/env python3
"""
Working YOLOv8 3D Dental Pipeline - Simplified Version
Uses second code projection + first code reprojection + YOLOv8 inference
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import json
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from plyfile import PlyData, PlyElement


class SimplePipeline:
    """Simplified but working pipeline."""

    def __init__(self):
        self.resolution = 512
        self.mesh_data = None
        self.projections = None
        self.inference_results = None
        self.yolo_model = None

    def load_ply_file(self, filename):
        """Load PLY file using second code method."""
        try:
            print(f"Loading PLY: {filename}")
            plydata = PlyData.read(filename)
            vertex_data = plydata['vertex']
            vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

            colors = None
            if 'red' in vertex_data and 'green' in vertex_data and 'blue' in vertex_data:
                colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T

            self.mesh_data = {'vertices': vertices, 'colors': colors, 'filename': filename}
            print(f"✅ Loaded {len(vertices)} vertices")
            return True
        except Exception as e:
            print(f"❌ PLY loading error: {e}")
            return False

    def load_yolo_model(self, model_path):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model: {model_path}")
            self.yolo_model = YOLO(model_path)
            print("✅ YOLO model loaded")
            return True
        except Exception as e:
            print(f"❌ YOLO loading error: {e}")
            return False

    def create_projections(self):
        """Create 2D projections using second code method."""
        if self.mesh_data is None:
            return False

        print("Creating projections...")
        vertices = self.mesh_data['vertices']
        colors = self.mesh_data['colors']

        # Store projections
        self.projections = {}

        # Occlusal projection
        self.projections['occlusal'] = self._create_occlusal(vertices, colors)

        # Panoramic projection
        self.projections['panoramic'] = self._create_panoramic(vertices, colors)

        # Buccal projections
        center_x = np.mean(vertices[:, 0])
        left_mask = vertices[:, 0] < center_x
        right_mask = vertices[:, 0] >= center_x

        self.projections['buccal_left'] = self._create_buccal(vertices[left_mask],
                                                              colors[left_mask] if colors is not None else None,
                                                              left_mask)
        self.projections['buccal_right'] = self._create_buccal(vertices[right_mask],
                                                               colors[right_mask] if colors is not None else None,
                                                               right_mask)

        print("✅ Projections created")
        return True

    def _create_occlusal(self, vertices, colors):
        """Create occlusal projection."""
        # Project to X-Z plane (top-down)
        x_coords = vertices[:, 0]
        z_coords = vertices[:, 2]
        y_coords = vertices[:, 1]  # height for depth

        # Normalize to image coordinates
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)

        if x_max == x_min or z_max == z_min:
            return None

        x_norm = ((x_coords - x_min) / (x_max - x_min) * (self.resolution - 1)).astype(int)
        z_norm = ((z_coords - z_min) / (z_max - z_min) * (self.resolution - 1)).astype(int)

        # Create image
        image = np.zeros((self.resolution, self.resolution, 3))
        vertex_map = {}

        for idx, (x, z, y) in enumerate(zip(x_norm, z_norm, y_coords)):
            if 0 <= x < self.resolution and 0 <= z < self.resolution:
                if colors is not None:
                    image[z, x] = colors[idx] / 255.0 if np.max(colors[idx]) > 1 else colors[idx]
                else:
                    image[z, x] = [0.8, 0.8, 0.8]  # Gray default

                # Store vertex mapping
                if (z, x) not in vertex_map:
                    vertex_map[(z, x)] = []
                vertex_map[(z, x)].append(idx)

        return {'image': image, 'vertex_map': vertex_map, 'type': 'occlusal'}

    def _create_panoramic(self, vertices, colors):
        """Create panoramic projection."""
        # Simple arch unwrapping - use X coordinate as arc position
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]  # height

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        if x_max == x_min or y_max == y_min:
            return None

        u_coords = ((x_coords - x_min) / (x_max - x_min) * (self.resolution - 1)).astype(int)
        v_coords = ((y_coords - y_min) / (y_max - y_min) * (self.resolution - 1)).astype(int)

        image = np.zeros((self.resolution, self.resolution, 3))
        vertex_map = {}

        for idx, (u, v) in enumerate(zip(u_coords, v_coords)):
            if 0 <= u < self.resolution and 0 <= v < self.resolution:
                if colors is not None:
                    image[v, u] = colors[idx] / 255.0 if np.max(colors[idx]) > 1 else colors[idx]
                else:
                    image[v, u] = [0.8, 0.8, 0.8]

                if (v, u) not in vertex_map:
                    vertex_map[(v, u)] = []
                vertex_map[(v, u)].append(idx)

        return {'image': image, 'vertex_map': vertex_map, 'type': 'panoramic'}

    def _create_buccal(self, vertices, colors, original_mask):
        """Create buccal projection."""
        if len(vertices) == 0:
            return None

        # Project to Y-Z plane (side view)
        y_coords = vertices[:, 1]  # height
        z_coords = vertices[:, 2]  # front-back

        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)

        if y_max == y_min or z_max == z_min:
            return None

        u_coords = ((z_coords - z_min) / (z_max - z_min) * (self.resolution - 1)).astype(int)
        v_coords = ((y_coords - y_min) / (y_max - y_min) * (self.resolution - 1)).astype(int)

        image = np.zeros((self.resolution, self.resolution, 3))
        vertex_map = {}

        # Get original indices
        original_indices = np.where(original_mask)[0]

        for i, (u, v) in enumerate(zip(u_coords, v_coords)):
            if 0 <= u < self.resolution and 0 <= v < self.resolution:
                if colors is not None:
                    image[v, u] = colors[i] / 255.0 if np.max(colors[i]) > 1 else colors[i]
                else:
                    image[v, u] = [0.8, 0.8, 0.8]

                if (v, u) not in vertex_map:
                    vertex_map[(v, u)] = []
                vertex_map[(v, u)].append(original_indices[i])

        return {'image': image, 'vertex_map': vertex_map, 'type': 'buccal'}

    def run_inference(self, confidence=0.5):
        """Run YOLO inference on all projections."""
        if self.yolo_model is None or self.projections is None:
            return False

        print("Running YOLO inference...")
        self.inference_results = {}

        for proj_name, proj_data in self.projections.items():
            if proj_data is None:
                continue

            print(f"  Processing {proj_name}...")

            # Convert image to proper format
            image = proj_data['image']
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Run inference
            try:
                results = self.yolo_model(image, conf=confidence, verbose=False)

                detections = []
                if results and len(results) > 0:
                    result = results[0]

                    if result.masks is not None:
                        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
                        classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
                        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

                        for i, (box, mask, cls, conf) in enumerate(zip(boxes, masks, classes, confidences)):
                            detection = {
                                'bbox': box,
                                'mask': mask,
                                'class_id': int(cls),
                                'confidence': float(conf),
                                'tooth_id': int(cls) + 1
                            }
                            detections.append(detection)

                self.inference_results[proj_name] = detections
                print(f"    Found {len(detections)} detections")

            except Exception as e:
                print(f"    Error in {proj_name}: {e}")
                self.inference_results[proj_name] = []

        total_detections = sum(len(dets) for dets in self.inference_results.values())
        print(f"✅ Total detections: {total_detections}")
        return True

    def reproject_to_3d(self):
        """Reproject results to 3D using first code method."""
        if self.inference_results is None or self.mesh_data is None:
            return False

        print("Reprojecting to 3D...")
        vertices = self.mesh_data['vertices']
        colors = self.mesh_data['colors']

        # Initialize result colors
        if colors is not None:
            result_colors = colors.copy()
        else:
            result_colors = np.full((len(vertices), 3), 128, dtype=np.uint8)

        # Tooth colors
        tooth_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 255, 0],
            [255, 0, 128], [128, 0, 255], [0, 128, 255], [255, 128, 128],
            [128, 255, 128], [128, 128, 255], [255, 255, 128], [255, 128, 255],
            [128, 255, 255], [192, 64, 64], [64, 192, 64], [64, 64, 192],
            [192, 192, 64], [192, 64, 192], [64, 192, 192], [255, 96, 96],
            [96, 255, 96], [96, 96, 255], [255, 192, 96], [96, 255, 192],
            [192, 96, 255], [255, 96, 192], [192, 255, 96], [96, 192, 255]
        ]

        tooth_assignments = {}

        # Process each projection's results
        for proj_name, detections in self.inference_results.items():
            proj_data = self.projections[proj_name]
            if proj_data is None:
                continue

            vertex_map = proj_data['vertex_map']

            for detection in detections:
                tooth_id = detection['tooth_id']
                mask_2d = detection['mask']

                # Resize mask if needed
                if mask_2d.shape != (512, 512):
                    mask_2d = cv2.resize(mask_2d.astype(np.uint8), (512, 512)) / 255.0

                # Get vertex indices from mask
                vertex_indices = set()
                mask_coords = np.where(mask_2d > 0.5)

                for y, x in zip(mask_coords[0], mask_coords[1]):
                    pixel_key = (int(y), int(x))
                    if pixel_key in vertex_map:
                        vertex_indices.update(vertex_map[pixel_key])

                if vertex_indices:
                    if tooth_id not in tooth_assignments:
                        tooth_assignments[tooth_id] = set()
                    tooth_assignments[tooth_id].update(vertex_indices)

        # Apply colors
        for tooth_id, vertex_indices in tooth_assignments.items():
            color = tooth_colors[(tooth_id - 1) % len(tooth_colors)]
            for vertex_idx in vertex_indices:
                if 0 <= vertex_idx < len(vertices):
                    result_colors[vertex_idx] = color

        self.final_results = {
            'vertices': vertices,
            'colors': result_colors,
            'tooth_assignments': tooth_assignments
        }

        print(f"✅ Reprojected {len(tooth_assignments)} teeth")
        return True

    def save_results(self, output_path):
        """Save 3D results."""
        if not hasattr(self, 'final_results'):
            return False

        os.makedirs(output_path, exist_ok=True)

        # Save JSON
        results_file = os.path.join(output_path, "results_3d.json")
        results_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_teeth': len(self.final_results['tooth_assignments']),
            'teeth_results': {}
        }

        for tooth_id, vertex_indices in self.final_results['tooth_assignments'].items():
            results_data['teeth_results'][str(tooth_id)] = {
                'vertex_count': len(vertex_indices),
                'vertex_indices': [int(idx) for idx in vertex_indices]
            }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save PLY
        ply_file = os.path.join(output_path, "segmented_mesh.ply")
        vertices = self.final_results['vertices']
        colors = self.final_results['colors']

        vertex_array = np.array([
            (vertices[i][0], vertices[i][1], vertices[i][2],
             colors[i][0], colors[i][1], colors[i][2])
            for i in range(len(vertices))
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        PlyData([vertex_element]).write(ply_file)

        print(f"✅ Results saved to {output_path}")
        return True


class SimpleGUI:
    """Simple working GUI."""

    def __init__(self, root):
        self.root = root
        self.pipeline = SimplePipeline()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("YOLOv8 3D Pipeline - Working Version")
        self.root.geometry("700x600")

        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="🦷 YOLOv8 3D Dental Pipeline",
                  font=('Arial', 16, 'bold')).pack(pady=(0, 20))

        # File inputs
        file_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 20))

        # PLY file
        ply_frame = ttk.Frame(file_frame)
        ply_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(ply_frame, text="PLY File:").pack(side=tk.LEFT)
        self.ply_path = tk.StringVar()
        ttk.Entry(ply_frame, textvariable=self.ply_path, width=40).pack(side=tk.LEFT, padx=(10, 10), fill=tk.X,
                                                                        expand=True)
        ttk.Button(ply_frame, text="Browse", command=self.browse_ply).pack(side=tk.RIGHT)

        # YOLO model
        model_frame = ttk.Frame(file_frame)
        model_frame.pack(fill=tk.X)
        ttk.Label(model_frame, text="YOLO Model:").pack(side=tk.LEFT)
        self.model_path = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_path, width=40).pack(side=tk.LEFT, padx=(10, 10), fill=tk.X,
                                                                            expand=True)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.RIGHT)

        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 20))

        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X)
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence = tk.DoubleVar(value=0.5)
        tk.Scale(conf_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.confidence).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Button(button_frame, text="🚀 Run Complete Pipeline",
                   command=self.run_pipeline, width=30).pack(pady=5)

        ttk.Button(button_frame, text="💾 Save Results",
                   command=self.save_results, width=30).pack(pady=5)

        # Status
        self.status = tk.StringVar(value="Ready - Select PLY file and YOLO model")
        ttk.Label(main_frame, textvariable=self.status, relief=tk.SUNKEN).pack(fill=tk.X)

        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))

        self.results_text = tk.Text(results_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Add message to results"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()

    def browse_ply(self):
        filename = filedialog.askopenfilename(
            title="Select PLY File",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if filename:
            self.ply_path.set(filename)

    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO models", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)

    def run_pipeline(self):
        """Run the complete pipeline."""
        ply_file = self.ply_path.get().strip()
        model_file = self.model_path.get().strip()

        if not ply_file or not model_file:
            messagebox.showerror("Error", "Please select both PLY file and YOLO model")
            return

        self.results_text.delete(1.0, tk.END)

        try:
            # Step 1: Load files
            self.status.set("Loading files...")
            self.log("🔄 Step 1: Loading files...")

            if not self.pipeline.load_ply_file(ply_file):
                messagebox.showerror("Error", "Failed to load PLY file")
                return
            self.log("✅ PLY file loaded")

            if not self.pipeline.load_yolo_model(model_file):
                messagebox.showerror("Error", "Failed to load YOLO model")
                return
            self.log("✅ YOLO model loaded")

            # Step 2: Create projections
            self.status.set("Creating projections...")
            self.log("\n🔄 Step 2: Creating projections...")

            if not self.pipeline.create_projections():
                messagebox.showerror("Error", "Failed to create projections")
                return
            self.log("✅ Projections created")

            # Step 3: Run inference
            self.status.set("Running inference...")
            self.log("\n🔄 Step 3: Running YOLO inference...")

            confidence = self.confidence.get()
            if not self.pipeline.run_inference(confidence):
                messagebox.showerror("Error", "Failed to run inference")
                return
            self.log("✅ Inference completed")

            # Step 4: Reproject to 3D
            self.status.set("Reprojecting to 3D...")
            self.log("\n🔄 Step 4: Reprojecting to 3D...")

            if not self.pipeline.reproject_to_3d():
                messagebox.showerror("Error", "Failed to reproject to 3D")
                return
            self.log("✅ 3D reprojection completed")

            # Show results
            total_teeth = len(self.pipeline.final_results['tooth_assignments'])
            total_vertices = sum(len(v) for v in self.pipeline.final_results['tooth_assignments'].values())

            self.log(f"\n📊 RESULTS:")
            self.log(f"   Detected teeth: {total_teeth}")
            self.log(f"   Segmented vertices: {total_vertices:,}")

            self.status.set("Pipeline completed successfully!")
            messagebox.showinfo("Success",
                                f"Pipeline completed!\n\nDetected {total_teeth} teeth\nSegmented {total_vertices:,} vertices")

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.log(f"❌ {error_msg}")
            self.status.set("Pipeline failed")
            messagebox.showerror("Error", error_msg)

    def save_results(self):
        """Save results to folder."""
        if not hasattr(self.pipeline, 'final_results'):
            messagebox.showwarning("Warning", "No results to save. Run pipeline first.")
            return

        output_path = filedialog.askdirectory(title="Select Output Folder")
        if output_path:
            if self.pipeline.save_results(output_path):
                self.log(f"\n💾 Results saved to: {output_path}")
                messagebox.showinfo("Saved", f"Results saved to:\n{output_path}")

                # Ask to open folder
                if messagebox.askyesno("Open Folder", "Open output folder?"):
                    try:
                        if os.name == 'nt':
                            os.startfile(output_path)
                        else:
                            os.system(f'open "{output_path}"')
                    except:
                        pass


def main():
    """Main function."""
    print("🚀 Starting YOLOv8 3D Pipeline...")

    root = tk.Tk()
    app = SimpleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()