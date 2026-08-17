#!/usr/bin/env python3

# PLY file utility functions for extracting and analyzing vertex data



import numpy as np

from plyfile import PlyData

import matplotlib



matplotlib.use('Qt5Agg')  # Use Qt5 backend for matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D





def extract_vertices_with_color(ply_filename):

    """

    Extract vertices with color information from a PLY file.

    Returns vertices as Nx3 array and colors as Nx3 array.



    Parameters:

    -----------

    ply_filename : str

        Path to the PLY file



    Returns:

    --------

    vertices : numpy.ndarray

        Nx3 array of vertex coordinates (x, y, z)

    colors : numpy.ndarray or None

        Nx3 array of RGB color values (red, green, blue)

    has_color : bool

        Whether the PLY file has color information

    """

    try:

        plydata = PlyData.read(ply_filename)

        vertex_data = plydata['vertex']



        # Extract vertices

        vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T



        # Extract colors if available

        has_color = False

        if 'red' in vertex_data and 'green' in vertex_data and 'blue' in vertex_data:

            colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T

            has_color = True

        else:

            colors = None



        return vertices, colors, has_color



    except Exception as e:

        print(f"Error extracting vertices: {e}")

        return None, None, False





def detect_tartar_by_pink_color(colors1, colors2):

    """

    Identify tartar areas by comparing "after tablet" (with pink) to "before tablet" models.

    This detects pixels that became pink in the second model.



    Parameters:

    -----------

    colors1 : numpy.ndarray

        RGB colors from first model (before tablet)

    colors2 : numpy.ndarray

        RGB colors from second model (after tablet, with pink areas)



    Returns:

    --------

    is_tartar : numpy.ndarray

        Boolean mask indicating which vertices are tartar areas

    """

    if colors1 is None or colors2 is None:

        return None



    # Pink color typically has high red, moderate green, and lower blue values

    # Define pink color range that represents tartar in "after tablet" model

    pink_min = np.array([180, 80, 80])

    pink_max = np.array([255, 180, 180])



    # Check which points became pink in the second model (after tablet)

    is_pink_in_model2 = ((colors2[:, 0] >= pink_min[0]) & (colors2[:, 0] <= pink_max[0]) &

                         (colors2[:, 1] >= pink_min[1]) & (colors2[:, 1] <= pink_max[1]) &

                         (colors2[:, 2] >= pink_min[2]) & (colors2[:, 2] <= pink_max[2]) &

                         (colors2[:, 0] > colors2[:, 2] + 20))  # Red channel significantly higher than blue



    # Additional check: points that weren't pink in the first model

    # This checks for significant color change to pink

    not_pink_in_model1 = ~((colors1[:, 0] >= pink_min[0]) & (colors1[:, 0] <= pink_max[0]) &

                           (colors1[:, 1] >= pink_min[1]) & (colors1[:, 1] <= pink_max[1]) &

                           (colors1[:, 2] >= pink_min[2]) & (colors1[:, 2] <= pink_max[2]) &

                           (colors1[:, 0] > colors1[:, 2] + 20))



    # Points that became pink in the second model are tartar

    is_tartar = is_pink_in_model2 & not_pink_in_model1



    return is_tartar





def visualize_tartar_detection(vertices, colors_before, colors_after, tartar_mask):

    """

    Visualize the before and after tablet models with tartar highlighted.



    Parameters:

    -----------

    vertices : numpy.ndarray

        Nx3 array of vertex coordinates

    colors_before : numpy.ndarray

        Nx3 array of RGB color values from before tablet model

    colors_after : numpy.ndarray

        Nx3 array of RGB color values from after tablet model

    tartar_mask : numpy.ndarray

        Boolean mask indicating which vertices are tartar



    Returns:

    --------

    fig : matplotlib.figure.Figure

        The created figure with three subplots

    """

    # Downsample data if too many points for visualization (more than 100,000)

    if len(vertices) > 100000:

        step = len(vertices) // 100000 + 1

        vertices_subset = vertices[::step]

        colors_before_subset = colors_before[::step]

        colors_after_subset = colors_after[::step]

        tartar_mask_subset = tartar_mask[::step]

        print(f"Downsampling from {len(vertices)} to {len(vertices_subset)} points for visualization")

    else:

        vertices_subset = vertices

        colors_before_subset = colors_before

        colors_after_subset = colors_after

        tartar_mask_subset = tartar_mask



    # Close any existing plots to free memory

    plt.close('all')



    fig = plt.figure(figsize=(18, 6))



    # Plot before tablet model

    ax1 = fig.add_subplot(131, projection='3d')

    normalized_colors_before = colors_before_subset / 255.0 if np.max(

        colors_before_subset) > 1.0 else colors_before_subset

    ax1.scatter(

        vertices_subset[:, 0], vertices_subset[:, 1], vertices_subset[:, 2],

        c=normalized_colors_before, s=1, marker='o'

    )

    ax1.set_title("Before Tablet")



    # Plot after tablet model (with pink areas)

    ax2 = fig.add_subplot(132, projection='3d')

    normalized_colors_after = colors_after_subset / 255.0 if np.max(colors_after_subset) > 1.0 else colors_after_subset

    ax2.scatter(

        vertices_subset[:, 0], vertices_subset[:, 1], vertices_subset[:, 2],

        c=normalized_colors_after, s=1, marker='o'

    )

    ax2.set_title("After Tablet (Pink Areas = Tartar)")



    # Plot with tartar highlighted

    ax3 = fig.add_subplot(133, projection='3d')



    # Create highlight colors: tartar areas in red, rest in gray

    highlight_colors = np.ones((len(vertices_subset), 3)) * 0.7  # Gray

    highlight_colors[tartar_mask_subset] = [1.0, 0.0, 0.0]  # Red for tartar



    ax3.scatter(

        vertices_subset[:, 0], vertices_subset[:, 1], vertices_subset[:, 2],

        c=highlight_colors, s=2, marker='o'  # Make tartar points slightly larger

    )

    ax3.set_title("Tartar Areas Highlighted")



    # Set consistent viewing angle and scale

    for ax in [ax1, ax2, ax3]:

        ax.set_xlabel('X')

        ax.set_ylabel('Y')

        ax.set_zlabel('Z')



        # Auto-scale to the data

        max_range = np.max([

            vertices_subset[:, 0].max() - vertices_subset[:, 0].min(),

            vertices_subset[:, 1].max() - vertices_subset[:, 1].min(),

            vertices_subset[:, 2].max() - vertices_subset[:, 2].min()

        ])



        mid_x = (vertices_subset[:, 0].max() + vertices_subset[:, 0].min()) / 2

        mid_y = (vertices_subset[:, 1].max() + vertices_subset[:, 1].min()) / 2

        mid_z = (vertices_subset[:, 2].max() + vertices_subset[:, 2].min()) / 2



        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)

        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)

        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)



    plt.tight_layout()

    return fig





def export_tartar_points(vertices, colors, is_tartar, filename):

    """

    Export the tartar points to a CSV file.



    Parameters:

    -----------

    vertices : numpy.ndarray

        Nx3 array of vertex coordinates

    colors : numpy.ndarray

        Nx3 array of RGB color values (typically from after-tablet model)

    is_tartar : numpy.ndarray

        Boolean mask indicating which vertices are tartar

    filename : str

        Path to save the CSV file



    Returns:

    --------

    num_points : int

        Number of tartar points exported

    """

    tartar_vertices = vertices[is_tartar]

    tartar_colors = colors[is_tartar]



    # Create a structured array with vertices and colors

    data = np.column_stack((tartar_vertices, tartar_colors))



    # Export to CSV

    header = "x,y,z,red,green,blue"

    np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%.6f,%.6f,%.6f,%d,%d,%d')



    return len(tartar_vertices)





def visualize_point_cloud(vertices, colors=None, title="Point Cloud Visualization"):

    """

    Visualize a 3D point cloud with optional color information.



    Parameters:

    -----------

    vertices : numpy.ndarray

        Nx3 array of vertex coordinates

    colors : numpy.ndarray or None

        Nx3 array of RGB color values

    title : str

        Title for the visualization



    Returns:

    --------

    fig : matplotlib.figure.Figure

        The created figure

    ax : matplotlib.axes.Axes

        The created 3D axis

    """

    # Downsample if too many points

    if len(vertices) > 100000:

        step = len(vertices) // 100000 + 1

        vertices = vertices[::step]

        if colors is not None:

            colors = colors[::step]

        print(f"Downsampled to {len(vertices)} points for visualization")



    plt.close('all')

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection='3d')



    # Normalize colors if they're in 0-255 range

    if colors is not None:

        normalized_colors = colors / 255.0 if np.max(colors) > 1.0 else colors

    else:

        normalized_colors = None



    # Plot the points

    if normalized_colors is not None:

        scatter = ax.scatter(

            vertices[:, 0], vertices[:, 1], vertices[:, 2],

            c=normalized_colors, s=1, marker='o'

        )

    else:

        scatter = ax.scatter(

            vertices[:, 0], vertices[:, 1], vertices[:, 2],

            s=1, marker='o'

        )



    # Set labels and title

    ax.set_xlabel('X')

    ax.set_ylabel('Y')

    ax.set_zlabel('Z')

    ax.set_title(title)



    # Auto-scale to the data

    max_range = np.max([

        vertices[:, 0].max() - vertices[:, 0].min(),

        vertices[:, 1].max() - vertices[:, 1].min(),

        vertices[:, 2].max() - vertices[:, 2].min()

    ])



    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) / 2

    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) / 2

    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) / 2



    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)

    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)

    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)



    plt.tight_layout()

    return fig, ax





def align_dental_models(vertices1, colors1, vertices2, colors2):

    """

    Align and match points between two dental models with different vertex counts.

    Uses nearest neighbor matching to find corresponding points.



    Parameters:

    -----------

    vertices1 : numpy.ndarray

        Nx3 array of vertex coordinates from model 1

    colors1 : numpy.ndarray

        Nx3 array of RGB color values from model 1

    vertices2 : numpy.ndarray

        Mx3 array of vertex coordinates from model 2

    colors2 : numpy.ndarray

        Mx3 array of RGB color values from model 2



    Returns:

    --------

    vertices_aligned : numpy.ndarray

        Nx3 array of aligned vertices (using model1's vertex positions)

    colors1_aligned : numpy.ndarray

        Nx3 array of model1's colors

    colors2_aligned : numpy.ndarray

        Nx3 array of model2's colors matched to model1's vertices

    """

    from scipy.spatial import cKDTree



    # Build KD-Tree for efficient nearest neighbor search

    tree2 = cKDTree(vertices2)



    # For each vertex in model 1, find the nearest neighbor in model 2

    distances, indices = tree2.query(vertices1, k=1)



    # Get the colors from model 2 that correspond to the nearest vertices

    colors2_aligned = colors2[indices]



    # Use model1's vertices and both color sets

    return vertices1, colors1, colors2_aligned