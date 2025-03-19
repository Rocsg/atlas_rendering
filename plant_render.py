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
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
from data_utils import get_curve,get_radius,read_csv_files,read_information_file,process_curves
from utils import create_color_palettes,get_color_from_z_roots,get_color_from_z_leaves,count_vertices_and_triangles,compute_z_min_max
from render_utils import SurfaceRenderer,SliceInteractionHandler,VolumeRenderer,setup_vtk_renderer,create_sphere_actor,create_tube_actor


def load_config(config_path="config.json"):
    """Charge la configuration depuis un fichier JSON."""
    with open(config_path, "r") as f:
        return json.load(f)


def setup_interaction(renderer, surface_renderer,volume_renderer):
    # Créer un gestionnaire d'événements pour les touches fléchées
    interaction_handler = SliceInteractionHandler(renderer, surface_renderer,volume_renderer)
    
    # Créer un interactor et attacher le gestionnaire d'événements
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderer.GetRenderWindow())
    renderWindowInteractor.AddObserver('KeyPressEvent', interaction_handler.on_key_press)

    return renderWindowInteractor


def process_leaves(renderer, all_leaves_csv_files, curve_points_dict_leaves, information, config, z_min_leaves, z_max_leaves, palettes):
    """Génère la visualisation des feuilles en 3D."""
    all_points = []
    
    for i, (filename, dir_index) in enumerate(all_leaves_csv_files):
        levels = [level * config.pixel_size for level in information[i]]
        curve_data = curve_points_dict_leaves[filename]
        curve_points = curve_data[0]

        start_point = curve_points[0]
        x_s, y_s, z_s = [coord * config.pixel_size for coord in start_point]

        curve_color = get_color_from_z_leaves(curve_points[0][2], z_min_leaves, z_max_leaves, dir_index, palettes)
        
        # Ajout de la sphère pour le premier point
        sphereActor = create_sphere_actor(x_s, y_s, z_s, config.base_radius * config.pixel_size, curve_color)
        renderer.AddActor(sphereActor)

        # Ajout des tubes reliant les points
        for j, point in enumerate(curve_points[1:], start=1):
            x, y, z = [coord * config.pixel_size for coord in point]
            
            # Choix du rayon en fonction de la hauteur z
            if z < levels[0]:
                radius = config.leaf_small_radius * config.pixel_size
            else:
                radius = config.leaf_mid_radius * config.pixel_size if z < levels[1] else config.leaf_big_radius * config.pixel_size

            # Ajout du tube
            tubeActor = create_tube_actor(x_s, y_s, z_s, x, y, z, radius, curve_color)
            renderer.AddActor(tubeActor)

            x_s, y_s, z_s = x, y, z
            all_points.append([x, y, z])

    return all_points

def process_roots(curve_points_dict_roots, all_roots_csv_files, radius_points, config, z_min_roots, z_max_roots, renderer):
    """
    Traitement des racines en créant des sphères pour les points de départ et des tubes pour les courbes,
    puis en les ajoutant au renderer.

    :param curve_points_dict_roots: Dictionnaire contenant les courbes des racines.
    :param all_roots_csv_files: Liste des fichiers CSV des racines.
    :param radius_points: Liste des points de rayon associés à chaque courbe de racines.
    :param config: Configuration contenant les paramètres pour le rendu.
    :param z_min_roots: Valeur minimale de Z pour le mappage de couleurs.
    :param z_max_roots: Valeur maximale de Z pour le mappage de couleurs.
    :param renderer: L'instance du renderer pour ajouter les acteurs.
    :return: Liste des points traités pour les racines.
    """

    all_points = []

    # Traitement des racines
    for i, filename in enumerate(all_roots_csv_files):
        curve_points = curve_points_dict_roots[filename]
        radius_list = radius_points[i]
        curve_color = get_color_from_z_roots(curve_points[0][2], z_min_roots, z_max_roots)
        x_s, y_s, z_s = [coord * config.pixel_size for coord in curve_points[0]]

        for j, point in enumerate(curve_points):
            x, y, z = [coord * config.pixel_size for coord in point]
            if j == 0:
                # Création des sphères pour le premier point de la racine
                sphereSource = vtk.vtkSphereSource()
                sphereSource.SetCenter(x, y, z)
                radius = radius_list[j][1] * config.pixel_size
                sphereSource.SetRadius((radius + 5) / 2)
                sphereSource.Update()

                # Setup du mapper et acteur pour la sphère
                sphereMapper = vtk.vtkPolyDataMapper()
                sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
                sphereActor = vtk.vtkActor()
                sphereActor.SetMapper(sphereMapper)
                sphereActor.GetProperty().SetOpacity(1.0)
                sphereActor.GetProperty().SetColor(curve_color)
                sphereActor.GetProperty().SetInterpolationToFlat()

                # Ajouter l'acteur sphère au renderer
                renderer.AddActor(sphereActor)
            else:
                # Création des tubes pour les points suivants
                radius = radius_list[j][1] * config.pixel_size
                lineSource = vtk.vtkLineSource()
                lineSource.SetPoint1(x_s, y_s, z_s)
                lineSource.SetPoint2(x, y, z)

                # Tube filter pour la courbe
                tubeFilter = vtk.vtkTubeFilter()
                tubeFilter.SetInputConnection(lineSource.GetOutputPort())
                tubeFilter.SetRadius((radius - 5) / 2)
                tubeFilter.SetNumberOfSides(12)
                tubeFilter.Update()

                # Setup du mapper et acteur pour le tube
                tubeMapper = vtk.vtkPolyDataMapper()
                tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())
                tubeActor = vtk.vtkActor()
                tubeActor.SetMapper(tubeMapper)
                tubeActor.GetProperty().SetOpacity(1.0)
                tubeActor.GetProperty().SetColor(curve_color)
                tubeActor.GetProperty().SetInterpolationToFlat()

                # Ajouter l'acteur tube au renderer
                renderer.AddActor(tubeActor)

                # Mettre à jour les points de départ pour la prochaine itération
                x_s, y_s, z_s = x, y, z

            # Ajouter le point courant aux points traités
            all_points.append([x, y, z])

    return all_points


def main(config):
    """
    Renders a 3D visualization of plant roots based on a series of CSV files
    containing curve and radius information, and generates an iso-surface
    using Marching Cubes.
    """

    # Initialisation du renderer VTK existant
    renderer, renderWindow = setup_vtk_renderer()

    # Lecture des fichiers CSV
    all_leaves_csv_files = read_csv_files(config.csv_directories_leaves, config.path_to_test_data, track_index=True)
    all_roots_csv_files = read_csv_files(config.csv_directories_roots, config.path_to_test_data, track_index=False)

    # Lecture du fichier d'information
    information = read_information_file(config.path_to_test_data + os.sep + config.information_path)

    # Extraction des courbes
    all_z_leaves, curve_points_dict_leaves = process_curves(all_leaves_csv_files, config.csv_directories_leaves, config.path_to_test_data, config.curve_sample_leaf, track_index=True)
    all_z_roots, curve_points_dict_roots = process_curves(all_roots_csv_files, config.csv_directories_roots, config.path_to_test_data, config.curve_sample_root, track_index=False)

    # Calcul des min/max de Z pour le mapping de couleurs
    z_min_roots, z_max_roots = compute_z_min_max(all_z_roots)
    z_min_leaves, z_max_leaves = compute_z_min_max(all_z_leaves)

    # Création des palettes de couleurs
    palettes = create_color_palettes(len(config.csv_directories_leaves))

    # Génération de la visualisation des feuilles
    all_points = process_leaves(renderer, all_leaves_csv_files, curve_points_dict_leaves, information, config, z_min_leaves, z_max_leaves, palettes)

    # Processus des racines
    curve_lengths = [len(curve_points) for curve_points in curve_points_dict_roots.values()]
    radius_points = get_radius(config.path_to_test_data + os.sep + config.radius_path, curve_lengths)
    
    all_points.extend(process_roots(curve_points_dict_roots, all_roots_csv_files, radius_points, config, z_min_roots, z_max_roots, renderer))

    # Création de la grille 3D et remplissage avec la densité des points
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

    # Marching Cubes pour l'extraction de la surface iso
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(grid)
    marchingCubes.ComputeNormalsOn()
    marchingCubes.SetValue(0, 0.9)
    marchingCubes.Update()

    # Création du mapper et de l'acteur pour la surface iso
    isoMapper = vtk.vtkPolyDataMapper()
    isoMapper.SetInputConnection(marchingCubes.GetOutputPort())
    isoActor = vtk.vtkActor()
    isoActor.SetMapper(isoMapper)
    isoActor.GetProperty().SetInterpolationToPhong()

    # Ajouter l'acteur iso-surface au renderer
    renderer.AddActor(isoActor)

    # Rendu de la surface et du volume
    surface_renderer = SurfaceRenderer(config.path_to_test_data + os.sep + config.tiff_file, config.x_offset, config.y_offset, config.z_offset, config.pixel_size, 3, 0.95, renderer, surface_file=config.path_to_test_data + os.sep + config.surface_file)
    renderer.AddActor(surface_renderer.surface_actor)

    volume_renderer = VolumeRenderer(
        tiff_file=config.path_to_test_data + os.sep + config.tiff_file,
        x_offset=config.x_offset,
        y_offset=config.y_offset,
        z_offset=config.z_offset,
        pixel_size=config.pixel_size,
        opacity_factor=5  # Ajuster selon ton besoin
    )
    renderer.AddVolume(volume_renderer.get_volume())

    total_vertices, total_triangles = count_vertices_and_triangles(renderer)
    print(f"Total vertices: {total_vertices}")
    print(f"Total triangles: {total_triangles}")

    renderer.ResetCamera()

    # Ajuster la caméra
    camera = renderer.GetActiveCamera()
    camera.Zoom(1)
    center = np.mean(all_points_np, axis=0)
    camera.SetPosition(center[0], center[1] - 50000, center[2]-35000)
    camera.SetFocalPoint(center[0], center[1], center[2])
    camera.SetViewUp(0, 0, 1)
    camera.SetClippingRange(1000, 80000)

    # Définir la taille et afficher la scène
    renderWindow.SetSize(800, 800)

    # Lancer l'interaction avec le renderer déjà créé
    renderWindow.Render()

    # Ajout de l'interaction pour le contrôle de la scène
    renderWindowInteractor = setup_interaction(renderer, surface_renderer, volume_renderer)
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
    

