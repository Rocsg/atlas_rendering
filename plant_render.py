import os
from types import SimpleNamespace
import numpy as np
import json
import vtk
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
from data_utils import get_radius, read_csv_files, read_information_file, process_curves
from utils import create_color_palettes, get_color_from_z_roots, get_color_from_z_leaves, count_vertices_and_triangles, compute_z_min_max
from render_utils import SurfaceRenderer, SliceInteractionHandler, VolumeRenderer, setup_vtk_renderer, create_sphere_actor, create_tube_actor


def load_config(config_path="config.json"):
    """Loads the configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def setup_interaction(renderer, surface_renderer, volume_renderer):
    # Create an event handler for the arrow keys
    interaction_handler = SliceInteractionHandler(renderer, surface_renderer, volume_renderer)
    
    # Create an interactor and attach the event handler
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderer.GetRenderWindow())
    interaction_handler.create_sliders()
    interaction_handler.create_text()
    interaction_handler.create_button()
    renderWindowInteractor.AddObserver('KeyPressEvent', interaction_handler.on_key_press)

    return renderWindowInteractor


def process_leaves(renderer, all_leaves_csv_files, curve_points_dict_leaves, information, config, z_min_leaves, z_max_leaves, palettes):
    """Generates the 3D visualization of the leaves."""
    all_points = []
    levels_dict = {}
    for i, (filename, dir_index) in enumerate(all_leaves_csv_files):
        levels_dict[filename] = [level * config.pixel_size for level in information[i]]
    for i, (filename, dir_index) in enumerate(all_leaves_csv_files):
        levels = levels_dict[filename]
        curve_data = curve_points_dict_leaves[filename]
        curve_points = curve_data[0]

        start_point = curve_points[0]
        x_s, y_s, z_s = [coord * config.pixel_size for coord in start_point]

        curve_color = get_color_from_z_leaves(curve_points[0][2], z_min_leaves, z_max_leaves, dir_index, palettes)
        
        # Add a sphere for the first point
        sphereActor = create_sphere_actor(x_s, y_s, z_s, config.base_radius * config.pixel_size, curve_color)
        renderer.AddActor(sphereActor)

        # Add tubes connecting the points
        for j, point in enumerate(curve_points[1:], start=1):
            x, y, z = [coord * config.pixel_size for coord in point]
            
            # Choose the radius based on the z heightœœœ
            if z < levels[0]:
                radius = config.leaf_small_radius * config.pixel_size
            else:
                radius = config.leaf_mid_radius * config.pixel_size if z < levels[1] else config.leaf_big_radius * config.pixel_size

            # Add the tubeœ
            tubeActor = create_tube_actor(x_s, y_s, z_s, x, y, z, radius, curve_color)
            renderer.AddActor(tubeActor)

            x_s, y_s, z_s = x, y, z
            all_points.append([x, y, z])

    return all_points

def process_roots(curve_points_dict_roots, all_roots_csv_files, radius_points, config, z_min_roots, z_max_roots, renderer):
    """
    Process the roots by creating spheres for the starting points and tubes for the curves,
    then adding them to the renderer.

    :param curve_points_dict_roots: Dictionary containing the root curves.
    :param all_roots_csv_files: List of root CSV files.
    :param radius_points: List of radius points associated with each root curve.
    :param config: Configuration containing rendering parameters.
    :param z_min_roots: Minimum Z value for color mapping.
    :param z_max_roots: Maximum Z value for color mapping.
    :param renderer: The renderer instance for adding actors.
    :return: List of processed points for the roots.
    """

    all_points = []

    # Process the roots
    for i, filename in enumerate(all_roots_csv_files):
        curve_points = curve_points_dict_roots[filename]
        radius_list = radius_points[i]
        curve_color = get_color_from_z_roots(curve_points[0][2], z_min_roots, z_max_roots)
        x_s, y_s, z_s = [coord * config.pixel_size for coord in curve_points[0]]
        append = vtk.vtkAppendPolyData()
        for j, point in enumerate(curve_points):
            x, y, z = [coord * config.pixel_size for coord in point]
            if j == 0:
                # Create spheres for the first root point
                sphereSource = vtk.vtkSphereSource()
                sphereSource.SetCenter(x, y, z)
                radius = radius_list[j][1] * config.pixel_size
                sphereSource.SetRadius((radius + 5) / 2)
                sphereSource.Update()

                # Setup the mapper and actor for the sphere
                sphereMapper = vtk.vtkPolyDataMapper()
                sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
                sphereActor = vtk.vtkActor()
                sphereActor.SetMapper(sphereMapper)
                sphereActor.GetProperty().SetOpacity(1.0)
                sphereActor.GetProperty().SetColor(curve_color)
                sphereActor.GetProperty().SetInterpolationToFlat()
            

                # Add the sphere actor to the renderer
                renderer.AddActor(sphereActor)
            else:
                # Create tubes for the following points
                radius = radius_list[j][1] * config.pixel_size
                lineSource = vtk.vtkLineSource()
                lineSource.SetPoint1(x_s, y_s, z_s)
                lineSource.SetPoint2(x, y, z)

                # Tube filter for the curve
                tubeFilter = vtk.vtkTubeFilter()
                tubeFilter.SetInputConnection(lineSource.GetOutputPort())
                tubeFilter.SetRadius((radius - 5) / 2)
                tubeFilter.SetNumberOfSides(12)
                tubeFilter.Update()

                # Setup the mapper and actor for the tube
                tubeMapper = vtk.vtkPolyDataMapper()
                tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())
                tubeActor = vtk.vtkActor()
                tubeActor.SetMapper(tubeMapper)
                tubeActor.GetProperty().SetOpacity(1.0)
                tubeActor.GetProperty().SetColor(curve_color)
                tubeActor.GetProperty().SetInterpolationToFlat()

                # Add the tube actor to the renderer
                renderer.AddActor(tubeActor)

                # Update the starting points for the next iteration
                x_s, y_s, z_s = x, y, z

            # Add the current point to the processed points
            all_points.append([x, y, z])

    return all_points


def main(config):
    """
    Renders a 3D visualization of plant roots based on a series of CSV files
    containing curve and radius information, and generates an iso-surface
    using Marching Cubes.
    """

    # Initialize the existing VTK renderer
    renderer, renderWindow = setup_vtk_renderer()

    # Read the CSV files
    all_leaves_csv_files = read_csv_files(config.csv_directories_leaves, config.path_to_test_data, track_index=True)
    all_roots_csv_files = read_csv_files(config.csv_directories_roots, config.path_to_test_data, track_index=False)

    # Read the information file
    information = read_information_file(config.path_to_test_data + os.sep + config.information_path)

    # Extract the curves
    all_z_leaves, curve_points_dict_leaves = process_curves(all_leaves_csv_files, config.csv_directories_leaves, config.path_to_test_data, config.curve_sample_leaf, track_index=True)
    all_z_roots, curve_points_dict_roots = process_curves(all_roots_csv_files, config.csv_directories_roots, config.path_to_test_data, config.curve_sample_root, track_index=False)

    # Compute the min/max Z values for color mapping
    z_min_roots, z_max_roots = compute_z_min_max(all_z_roots)
    z_min_leaves, z_max_leaves = compute_z_min_max(all_z_leaves)

    # Create the color palettes
    palettes = create_color_palettes(len(config.csv_directories_leaves))

    # Generate the leaf visualization
    all_points = process_leaves(renderer, all_leaves_csv_files, curve_points_dict_leaves, information, config, z_min_leaves, z_max_leaves, palettes)

    # Process the roots
    curve_lengths = [len(curve_points) for curve_points in curve_points_dict_roots.values()]
    radius_points = get_radius(config.path_to_test_data + os.sep + config.radius_path, curve_lengths)
    
    all_points.extend(process_roots(curve_points_dict_roots, all_roots_csv_files, radius_points, config, z_min_roots, z_max_roots, renderer))
    all_points_np = np.array(all_points)

    # Rendering the surface and volume
    surface_renderer = SurfaceRenderer(config.path_to_test_data + os.sep + config.tiff_file, config.x_offset, config.y_offset, config.z_offset, config.pixel_size, 3, 0.95, renderer, surface_file=config.path_to_test_data + os.sep + config.surface_file)
    renderer.AddActor(surface_renderer.surface_actor)

    volume_renderer = VolumeRenderer(
        tiff_file=config.path_to_test_data + os.sep + config.tiff_file,
        x_offset=config.x_offset,
        y_offset=config.y_offset,
        z_offset=config.z_offset,
        pixel_size=config.pixel_size,
        opacity_factor=5  # Adjust as needed
    )
    renderer.AddVolume(volume_renderer.get_volume())

    total_vertices, total_triangles = count_vertices_and_triangles(renderer)
    print(f"Total vertices: {total_vertices}")
    print(f"Total triangles: {total_triangles}")

    renderer.ResetCamera()

    # Adjust the camera
    camera = renderer.GetActiveCamera()
    camera.Zoom(1)
    center = np.mean(all_points_np, axis=0)
    camera.SetPosition(center[0], center[1] - 50000, center[2] - 35000)
    camera.SetFocalPoint(center[0], center[1], center[2])
    camera.SetViewUp(0, 0, 1)
    camera.SetClippingRange(1000, 80000)

    # Set the window size and display the scene
    renderWindow.SetSize(800, 800)

    # Start interaction with the already created renderer
    renderWindow.Render()

    # Add interaction for scene control
    renderWindowInteractor = setup_interaction(renderer, surface_renderer, volume_renderer)
    renderWindowInteractor.Start()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a 3D plant model from CSV files.")
    parser.add_argument("config", type=str, nargs="?", default=None, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load the configuration from the JSON
    if(args.config is None):
        # Load from default file (i.e. config.json)
        config_obj = load_config()
    else:
        config_obj = load_config(args.config)
    config = SimpleNamespace(**config_obj)

    # Call the main function with the parameters from the JSON
    main(config)
