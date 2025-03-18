"""
This script renders a 3D plant model, including roots and leaves, using VTK based on CSV files.

CSV Files Required:
1. **Root Positions CSV**: Contains the positions of the center of each root at different slices. One root is a row.
2. **Leaf Positions CSV**: Contains the positions of leaves at different slices. One leaf is a row.
3. **Root Radius CSV**: Contains the radius values for roots at different slices, with each row representing a root. One root is a row.
4. **Leaf Change Information CSV**: Contains information about when leaf characteristics change across slices. One leaf is a row.

Additional Notes:
- The **position CSV** should correspond to the same TIF file for accurate results.
- The **radius CSV** values should ideally be in the same range as the Z-values of the root curves, although perfect alignment is not mandatory but recommended for better fidelity.

This code creates a 3D visualization of the plant by reading and processing these CSV files using VTK to render the geometry. 
The leaves from the same tiller will share a color palette, becoming darker the higher they start.
The roots will have a blue color if they start high, and the red one if they start low.
"""
import os
from types import SimpleNamespace
import numpy as np
import json
import vtk
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkCommonColor import vtkNamedColors

from utils import get_curve,get_information,create_color_palettes,get_radius,SliceInteractionHandler,SurfaceRenderer



def load_config(config_path="config.json"):
    """Charge la configuration depuis un fichier JSON."""
    with open(config_path, "r") as f:
        return json.load(f)

def count_vertices_and_triangles(renderer):
    total_vertices = 0
    total_triangles = 0
    
    # Iterate over all actors in the renderer
    actors = renderer.GetActors()
    actors.InitTraversal()
    
    # Loop through each actor in the scene
    actor = actors.GetNextItem()
    while actor:
        # Get the mapper of the actor
        mapper = actor.GetMapper()
        
        # Get the polydata from the mapper
        poly_data = mapper.GetInput()
        
        # Count the number of vertices and triangles
        total_vertices += poly_data.GetNumberOfPoints()
        total_triangles += poly_data.GetNumberOfCells()  # Assuming the polydata has cells like triangles
        
        # Move to the next actor
        actor = actors.GetNextItem()
    
    return total_vertices, total_triangles

def get_color_from_z_leaves(z, z_min, z_max, dir_index, palettes):
    """
    Chooses a color palette based on the dir_index and returns the color for z.
    """
    # Normalize z to the range [0, 1] and clip it to ensure it's within bounds
    normalized_z = np.clip(1 - (z - z_min) / (z_max - z_min), 0, 1)

    # Retrieve the color palette (lookup table) based on the direction index
    lookup_table = palettes[dir_index]
    
    # Pre-set table range for lookup table, which is constant
    lookup_table.SetTableRange(0.0, 1.0)

    # Convert the normalized z value to an integer index between 0 and 255
    color = lookup_table.GetTableValue(int(normalized_z * 255))[:3]

    return color



def get_color_from_z_roots(z, z_min, z_max):
    """
    Uses a "jet" color palette for greater color variation.
    """
    # Create a "jet" color table that varies from blue (low) to red (high)
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetNumberOfTableValues(256)
    lookup_table.Build()

    # Normalize z between 0 and 1
    normalized_z = (z - z_min) / (z_max - z_min)

    # Use the predefined "jet" palette in VTK (which does a full color gradient)
    lookup_table.SetTableRange(0.0, 1.0)

    # Get the RGB color from the jet palette
    color = lookup_table.GetTableValue(int(normalized_z * 255))[:3]

    return color

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Fonction pour calculer l'angle de rotation et l'axe de rotation
def calculate_rotation_vector(v):
    # Vecteur de direction du cylindre
    direction = np.array(v)
    
    # Vecteur référence (on suppose l'axe Z comme référence pour la rotation)
    reference = np.array([0.0, 0.0, 1.0])

    # Produit scalaire pour calculer l'angles
    dot_product = np.dot(reference, direction)
    angle = np.arccos(dot_product)

    # Produit vectoriel pour calculer l'axe de rotation
    axis = np.cross(reference, direction)
    
    # Normalisation de l'axe de rotation
    axis = normalize(axis)
    
    return axis, np.degrees(angle)



def setup_interaction(renderer, surface_renderer):
    # Créer un gestionnaire d'événements pour les touches fléchées
    interaction_handler = SliceInteractionHandler(renderer, surface_renderer)
    
    # Créer un interactor et attacher le gestionnaire d'événements
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderer.GetRenderWindow())
    renderWindowInteractor.AddObserver('KeyPressEvent', interaction_handler.on_key_press)

    return renderWindowInteractor


def main(config):
    """
    Renders a 3D visualization of plant roots based on a series of CSV files
    containing curve and radius information, and generates an iso-surface
    using Marching Cubes.
    """
    # Define directories for CSV files

    # Initialize file lists
    all_leaves_csv_files = []
    all_roots_csv_files = []

    # Read CSV files for leaves and roots
    for i, directory in enumerate(config.csv_directories_leaves):
        files = sorted(
            [filename for filename in os.listdir(config.path_to_test_data + os.sep + directory) if filename.endswith(".csv")],
            key=lambda x: x.lower()
        )
        all_leaves_csv_files += [[filename, i] for filename in files]

    for directory in config.csv_directories_roots:
        all_roots_csv_files += sorted(
            [filename for filename in os.listdir(config.path_to_test_data + os.sep + directory) if filename.endswith(".csv")],
            key=lambda x: x.lower()
        )

    # Get curve data for leaves and roots
    all_z_leaves = []
    curve_points_dict_leaves = {}
    for filename, i in all_leaves_csv_files:
        directory = config.csv_directories_leaves[i]
        csv_path = os.path.join(config.path_to_test_data + os.sep + directory, filename)
        curve_points = get_curve(csv_path,config.curve_sample_leaf)
        curve_points_dict_leaves[filename] = [curve_points, i]
        all_z_leaves.append(curve_points[0][2])
    n_dir = i + 1

    all_z_roots = []
    curve_points_dict_roots = {}
    for filename in all_roots_csv_files:
        for directory in config.csv_directories_roots:
            csv_path = os.path.join(config.path_to_test_data + os.sep + directory, filename)
            if os.path.exists(csv_path):
                break
        curve_points = get_curve(csv_path,config.curve_sample_root)
        curve_points_dict_roots[filename] = curve_points
        all_z_roots.append(curve_points[0][2])

    # Calculate min/max Z for color mapping
    z_min_roots = min(all_z_roots)
    z_max_roots = max(all_z_roots)
    palettes = create_color_palettes(n_dir)

    z_min_leaves = min(all_z_leaves)
    z_max_leaves = max(all_z_leaves)

    # Prepare VTK rendering environment
    colors = vtkNamedColors()
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('IsoSurface with Marching Cubes')
    renderWindow.AddRenderer(renderer)

    # Read information file
    information = get_information(config.path_to_test_data + os.sep + config.information_path)
    information = [[float(char) for char in sublist] for sublist in information]

    # Create objects at each point (leaves)
    all_points = []
    for i, (filename, dir_index) in enumerate(all_leaves_csv_files):
        levels = information[i]
        levels = [level * config.pixel_size for level in levels]
        curve_data = curve_points_dict_leaves[filename]
        curve_points = curve_data[0]
        start_point = curve_points[0]
        x_s, y_s, z_s = [coord * config.pixel_size for coord in start_point]
        curve_color = get_color_from_z_leaves(
            curve_points[0][2], z_min_leaves, z_max_leaves, dir_index, palettes
        )
        for j, point in enumerate(curve_points):
            x, y, z = [coord * config.pixel_size for coord in point]
            if j == 0:
                sphereSource = vtkSphereSource()
                sphereSource.SetCenter(x, y, z)
                radius = config.base_radius * config.pixel_size
                sphereSource.SetRadius(radius)
                sphereSource.Update()

                # Setup sphere mapper and actor
                sphereMapper = vtkPolyDataMapper()
                sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
                sphereActor = vtkActor()
                sphereActor.SetMapper(sphereMapper)
                sphereActor.GetProperty().SetOpacity(1.0)
                sphereActor.GetProperty().SetColor(curve_color)
                sphereActor.GetProperty().SetInterpolationToFlat()
                # Add the sphere actor to the renderer
                renderer.AddActor(sphereActor)
            else:
                if z < levels[0]:
                    radius = config.leaf_small_radius * config.pixel_size
                else : 
                    radius = config.leaf_mid_radius * config.pixel_size if z < levels[1] else config.leaf_big_radius * config.pixel_size
                
                # Tube
                lineSource = vtkLineSource()
                lineSource.SetPoint1(x_s, y_s, z_s)
                lineSource.SetPoint2(x, y, z)

                # Tube filter for the curve
                tubeFilter = vtkTubeFilter()
                tubeFilter.SetInputConnection(lineSource.GetOutputPort())
                tubeFilter.SetRadius(radius)
                tubeFilter.SetNumberOfSides(12)
                tubeFilter.Update()

                # Setup tube mapper and actor
                tubeMapper = vtkPolyDataMapper()
                tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())
                tubeActor = vtkActor()
                tubeActor.SetMapper(tubeMapper)
                tubeActor.GetProperty().SetOpacity(1.0)
                tubeActor.GetProperty().SetColor(curve_color)
                tubeActor.GetProperty().SetInterpolationToFlat()
                # Add the tube actor to the renderer
                renderer.AddActor(tubeActor)

                x_s, y_s, z_s = x, y, z
            all_points.append([x, y, z])

    # Process roots
    curve_lengths = [len(curve_points) for curve_points in curve_points_dict_roots.values()]
    radius_points = get_radius(config.path_to_test_data + os.sep + config.radius_path, curve_lengths)
    for i, filename in enumerate(all_roots_csv_files):
        curve_points = curve_points_dict_roots[filename]
        radius_list = radius_points[i]
        curve_color = get_color_from_z_roots(curve_points[0][2], z_min_roots, z_max_roots)
        x_s, y_s, z_s = [coord * config.pixel_size for coord in curve_points[0]]
        for j, point in enumerate(curve_points):
            x, y, z = [coord * config.pixel_size for coord in point]
            if j == 0:
                sphereSource = vtkSphereSource()
                sphereSource.SetCenter(x, y, z)
                radius = radius_list[j][1] * config.pixel_size
                sphereSource.SetRadius((radius+5)/2)
                sphereSource.Update()

                # Setup sphere mapper and actor
                sphereMapper = vtkPolyDataMapper()
                sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
                sphereActor = vtkActor()
                sphereActor.SetMapper(sphereMapper)
                sphereActor.GetProperty().SetOpacity(1.0)
                sphereActor.GetProperty().SetColor(curve_color)

                # Add the sphere actor to the renderer
                sphereActor.GetProperty().SetInterpolationToFlat()
                renderer.AddActor(sphereActor)
            else : 
                radius = radius_list[j][1] * config.pixel_size
                lineSource = vtkLineSource()
                lineSource.SetPoint1(x_s, y_s, z_s)
                lineSource.SetPoint2(x, y, z)

                # Tube filter for the curve
                tubeFilter = vtkTubeFilter()
                tubeFilter.SetInputConnection(lineSource.GetOutputPort())
                tubeFilter.SetRadius((radius-5)/2)
                tubeFilter.SetNumberOfSides(12)
                tubeFilter.Update()

                # Setup tube mapper and actor
                tubeMapper = vtkPolyDataMapper()
                tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())
                tubeActor = vtkActor()
                tubeActor.SetMapper(tubeMapper)
                tubeActor.GetProperty().SetOpacity(1.0)
                tubeActor.GetProperty().SetColor(curve_color)

                # Add the tube actor to the renderer
                tubeActor.GetProperty().SetInterpolationToFlat()
                renderer.AddActor(tubeActor)

                x_s, y_s, z_s = x, y, z



            all_points.append([x, y, z])

    # Create 3D grid and fill it with point density
    spacing = 2.0
    grid = vtk.vtkImageData()
    grid.SetDimensions(500, 500, 500)
    grid.SetSpacing(spacing, spacing, spacing)
    grid.AllocateScalars(vtk.VTK_FLOAT, 1)
    all_points_np = np.array(all_points)
    min_coords = all_points_np.min(axis=0)
    max_coords = all_points_np.max(axis=0)
    grid_size = np.array([500, 500, 500])
    scale = (grid_size - 1) / (max_coords - min_coords)
    print("min_coords:", min_coords)
    print("max_coords:", max_coords)
    print("scale:", scale)
    normalized_points = (all_points_np - min_coords) * scale
    normalized_points = np.clip(normalized_points, 0, grid_size - 1)

    # Fill grid with point density
    x_idx, y_idx, z_idx = np.round(normalized_points).astype(int).T
    for x, y, z in zip(x_idx, y_idx, z_idx):
        grid.SetScalarComponentFromDouble(x, y, z, 0, 1.0)

    # Use Marching Cubes for iso-surface extraction
    marchingCubes = vtkMarchingCubes()
    marchingCubes.SetInputData(grid)
    marchingCubes.ComputeNormalsOn()
    marchingCubes.SetValue(0, 0.9)
    marchingCubes.Update()

    # Create mapper and actor for iso-surface
    isoMapper = vtkPolyDataMapper()
    isoMapper.SetInputConnection(marchingCubes.GetOutputPort())
    isoActor = vtkActor()
    isoActor.SetMapper(isoMapper)
    isoActor.GetProperty().SetInterpolationToPhong()

    # Add iso-surface actor to renderer
    renderer.AddActor(isoActor)


    use_orthoslicer=True
    if(use_orthoslicer):
        surface_renderer=SurfaceRenderer(config.path_to_test_data + os.sep + config.tiff_file,480,94,1693,config.pixel_size,3,0.95,renderer,surface_file=config.path_to_test_data + os.sep + config.surface_file)
        renderer.AddActor(surface_renderer.surface_actor)
    else:
        a=1 #TODO
        #Faire pareil mais avec du volume rendering

    total_vertices, total_triangles = count_vertices_and_triangles(renderer)
    print(f"Total vertices: {total_vertices}")
    print(f"Total triangles: {total_triangles}")

    # Set up the render window and interaction
    renderWindowInteractor = setup_interaction(renderer, surface_renderer)
    renderer.SetBackground(colors.GetColor3d('DarkSlateGray'))
    renderer.ResetCamera()

    # Adjust camera
    camera = renderer.GetActiveCamera()
    camera.Zoom(1)
    center = np.mean(all_points_np, axis=0)
    camera.SetPosition(center[0], center[1] -40000, center[2])
    camera.SetFocalPoint(center[0], center[1], center[2])
    camera.SetViewUp(0, 0, 1)

    renderWindow.SetSize(800, 800)
    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a 3D plant model from CSV files.")
    parser.add_argument("config", type=str, nargs="?",default=None,help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Charger la configuration depuis le JSON
    if(args.config is None):
        #Load from default file (i.e. config.json)
        config_obj = load_config()
    else :
        config_obj = load_config(args.config)
    config = SimpleNamespace(**config_obj)

    # Appeler la fonction main avec les paramètres du JSON
    main(config)
        
    if(False):
        base_radius=config["base_radius"],
        leaf_small_radius=config["leaf_small_radius"],
        leaf_mid_radius=config["leaf_mid_radius"],
        leaf_big_radius=config["leaf_big_radius"],
        curve_sample_root=config["curve_sample_root"],
        curve_sample_leaf=config["curve_sample_leaf"],
        pixel_size=config["pixel_size"],
        csv_directories_leaves=config["csv_directories_leaves"],
        csv_directories_roots=config["csv_directories_roots"],
        information_path=config["information_path"],
        radius_path=config["radius_path"],
        surface_file=config["surface_file"],
        tiff_file=config["tiff_file"]
    

