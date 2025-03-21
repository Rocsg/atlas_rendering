import vtk
import os
import json
from types import SimpleNamespace
import numpy as np
from data_utils import read_csv_files, read_information_file, process_curves, get_radius 
from utils import compute_z_min_max,create_color_palettes,get_color_from_z_leaves,get_color_from_z_roots
from collections import deque


def load_config(config_path="config.json"):
    """Loads the configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)
    
def generate_cylinder(point_start, point_end, radius, number_of_sides):
    all_points = []
    
    # Calcul du vecteur directeur du cylindre
    vector = np.array(point_end) - np.array(point_start)
    length = np.linalg.norm(vector)
    direction = vector / length  # Direction normalisée
    
    # Calcul de la normale et de la binormale
    normal = np.cross(direction, [0, 0, 1])
    normal = normal / np.linalg.norm(normal)  # Normalisé
    binormal = np.cross(direction, normal)
    # Générer les points pour la base du cylindre (point_start)
    for i in range(number_of_sides):
        angle = 2 * np.pi * i / number_of_sides
        point_on_circle = point_start + radius * (np.cos(angle) * normal + np.sin(angle) * binormal)
        all_points.append(point_on_circle)
    # Générer les points pour l'extrémité du cylindre (point_end)
    for i in range(number_of_sides):
        angle = 2 * np.pi * i / number_of_sides
        point_on_circle = point_end + radius * (np.cos(angle) * normal + np.sin(angle) * binormal)
        all_points.append(point_on_circle)
    # Générer les points de la surface latérale
    for i in range(number_of_sides):
        for j in range(int(length)):  # On parcours la longueur du cylindre en divisant en segments
            z = point_start[2] + j * (length / int(length))  # Calcul de la position verticale (z)
            point_on_lateral_surface = point_start + direction * j + radius * (np.cos(2 * np.pi * i / number_of_sides) * normal + np.sin(2 * np.pi * i / number_of_sides) * binormal)
            all_points.append(point_on_lateral_surface)
    
    return all_points

def generate_sphere(point, radius, number_of_points=100):
    all_points = []
    u = np.linspace(0, 2 * np.pi, number_of_points)
    v = np.linspace(0, np.pi, number_of_points)

    for i in range(len(u)):
        for j in range(len(v)):
            x = point[0] + radius * np.cos(u[i]) * np.sin(v[j])
            y = point[1] + radius * np.sin(u[i]) * np.sin(v[j])
            z = point[2] + radius * np.cos(v[j])
            all_points.append([x, y, z])

    return all_points
def process_leaves(all_leaves_csv_files, curve_points_dict_leaves, information, config, z_min_leaves, z_max_leaves, palettes):
    """Generates the 3D points for the leaves (without rendering actors)."""
    all_points = []

    for i, (filename, dir_index) in enumerate(all_leaves_csv_files):
        print(filename)
        curve_color = get_color_from_z_leaves(curve_points[0][2], z_min_leaves, z_max_leaves, dir_index, palettes)
        levels = [level * config.pixel_size for level in information[i]]
        curve_data = curve_points_dict_leaves[filename]
        curve_points = curve_data[0]

        start_point = curve_points[0]
        x_s, y_s, z_s = [coord * config.pixel_size for coord in start_point]

        curve_color = get_color_from_z_leaves(curve_points[0][2], z_min_leaves, z_max_leaves, dir_index, palettes)
        # Add the first point (you can ignore adding actors here)
        all_points.append([x_s, y_s, z_s] + list(curve_color))  # Including color info
        sphere_points = generate_sphere([x_s, y_s, z_s], config.base_radius)
        for p in sphere_points:
            all_points.append([*p] + list(curve_color))
        for j, point in enumerate(curve_points[1:], start=1):
            x, y, z = [coord * config.pixel_size for coord in point]
            
            # Choose the radius based on the z height (though radius is not used for iso-surfalist(curvece)
            if z < levels[0]:
                radius = config.leaf_small_radius * config.pixel_size
            else:
                radius = config.leaf_mid_radius * config.pixel_size if z < levels[1] else config.leaf_big_radius * config.pixel_size

            # Store the point and its associated color
            
            all_points.append([x, y, z] + list(curve_color))  # Including color info
            cylinder_points = generate_cylinder([x_s, y_s, z_s], [x, y, z], radius, 6)
            for p in cylinder_points:
                all_points.append([*p] + list(curve_color))
            x_s, y_s, z_s = x, y, z

    return all_points


def process_roots(curve_points_dict_roots, all_roots_csv_files, radius_points, config, z_min_roots, z_max_roots):
    """Process the roots by generating points (without rendering actors)."""
    all_points = []

    # Process the roots
    for i, filename in enumerate(all_roots_csv_files):
        print(filename)
        curve_points = curve_points_dict_roots[filename]
        radius_list = radius_points[i]
        curve_color = get_color_from_z_roots(curve_points[0][2], z_min_roots, z_max_roots)
        print(curve_color)
        x_s, y_s, z_s = [coord * config.pixel_size for coord in curve_points[0]]
        for j, point in enumerate(curve_points):
            x, y, z = [coord * config.pixel_size for coord in point]
            radius = radius_list[j][1] * config.pixel_size
            # Add the first point with its color
            if j == 0:
                all_points.append([x, y, z] + list(curve_color))  # Including color info
                sphere_points = generate_sphere([x_s, y_s, z_s], config.base_radius)
                for p in sphere_points:
                    all_points.append([*p] + list(curve_color))
            else:
                # For the other points, we continue adding the points and their color
                all_points.append([x, y, z] + list(curve_color))  # Including color info
                cylinder_points = generate_cylinder([x_s, y_s, z_s], [x, y, z], radius, 6)
                for p in cylinder_points:
                    all_points.append([*p] + list(curve_color))

            # Update the starting points for the next iteration
            x_s, y_s, z_s = x, y, z

    return all_points

def is_valid(x, y, z,grid_size):
    """Vérifier si les indices sont valides dans la grille"""
    return 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]

def propagate_colors(color_grid,grid_size,directions):
    """Propager les couleurs de manière optimisée dans la grille"""
    # Créer une file pour stocker les points à vérifier (points avec des couleurs assignées)
    queue = deque()

    # Tableau des visites (évite de traiter plusieurs fois le même point)
    visited = np.zeros((grid_size[0], grid_size[1], grid_size[2]), dtype=bool)

    # Ajouter les points de départ (celles avec une couleur assignée)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                if np.any(color_grid[x, y, z] != 0):  # Si la cellule n'est pas noire (c'est-à-dire assignée)
                    queue.append((x, y, z))
                    visited[x, y, z] = True  # Marquer comme visité

    # Propagation des couleurs
    while queue:
        x, y, z = queue.popleft()

        # Vérifier les voisins
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz

            if is_valid(nx, ny, nz) and not visited[nx, ny, nz]:  # Si le voisin n'a pas été visité
                if np.all(color_grid[nx, ny, nz] == 0):  # Si le voisin n'a pas de couleur
                    color_grid[nx, ny, nz] = color_grid[x, y, z]  # Assigner la couleur du voisin
                    queue.append((nx, ny, nz))  # Ajouter ce voisin à la file
                    visited[nx, ny, nz] = True  # Marquer comme visité

    return color_grid


def main(config):
    """
    Renders a 3D visualization of plant roots based on a series of CSV files
    contenant des courbes et informations de rayon, puis génère une iso-surface
    utilisant Marching Cubes.
    """

    print("Avant chargement données")
    # Lire les fichiers CSV pour les feuilles et racines (comme avant)
    all_leaves_csv_files = read_csv_files(config.csv_directories_leaves, config.path_to_test_data, track_index=True)
    all_roots_csv_files = read_csv_files(config.csv_directories_roots, config.path_to_test_data, track_index=False)

    # Lire les informations du fichier
    information = read_information_file(config.path_to_test_data + os.sep + config.information_path)

    # Processus des courbes pour obtenir les points des feuilles et racines
    all_z_leaves, curve_points_dict_leaves = process_curves(all_leaves_csv_files, config.csv_directories_leaves, config.path_to_test_data, config.curve_sample_leaf, track_index=True)
    all_z_roots, curve_points_dict_roots = process_curves(all_roots_csv_files, config.csv_directories_roots, config.path_to_test_data, config.curve_sample_root, track_index=False)

    # Calcul des min/max Z pour la colorisation
    z_min_roots, z_max_roots = compute_z_min_max(all_z_roots)
    z_min_leaves, z_max_leaves = compute_z_min_max(all_z_leaves)

    # Combiner les points des feuilles et des racines
    all_points = []
    palettes = create_color_palettes(len(config.csv_directories_leaves))
    # Processus des feuilles
    #all_points.extend(process_leaves(all_leaves_csv_files, curve_points_dict_leaves, information, config, z_min_leaves, z_max_leaves, palettes))

    # Processus des racines
    curve_lengths = [len(curve_points) for curve_points in curve_points_dict_roots.values()]
    radius_points = get_radius(config.path_to_test_data + os.sep + config.radius_path, curve_lengths)
    all_points.extend(process_roots(curve_points_dict_roots, all_roots_csv_files, radius_points, config, z_min_roots, z_max_roots))
    print("nombre de points : ",len(all_points))
    # Créer la grille 3D
    spacing = config.pixel_size
    grid = vtk.vtkImageData()
    grid.SetDimensions(500, 500, 500)
    grid.SetSpacing(1, 1, 1)
    grid.AllocateScalars(vtk.VTK_FLOAT, 1)
    
    all_points_np = np.array(all_points)
    print("shape np all points : ",all_points_np.shape)
    # Séparer les coordonnées et les couleurs
    coordinates = all_points_np[:, :3]  # Extraire les coordonnées (x, y, z)
    #print(coordinates[:100])
    colors = all_points_np[:, 3:]       # Extraire les couleurs
    unique_colors = set(map(tuple, colors))

    # Afficher les couleurs uniques
    print("Couleurs uniques trouvées :")
    for color in unique_colors:
        print(color)
    print("Nombre de couleurs :", len(colors))
    print("Nombre de coordonnées :",len(coordinates))
    
    # Calculer les min/max pour les coordonnées (pas les couleurs)
    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)
    min_coords_xyz = min_coords[:3]  
    max_coords_xyz = max_coords[:3]

    grid_size = np.array([500, 500, 500])
    scale = (grid_size - 1) / (max_coords_xyz - min_coords_xyz)
    
    # Normaliser les coordonnées
    normalized_coordinates = (coordinates - min_coords_xyz) * scale
    normalized_coordinates = np.round(normalized_coordinates, 3)
    normalized_coordinates = np.clip(normalized_coordinates, 0, grid_size - 1)
    normalized_coordinates_with_colors = np.hstack((normalized_coordinates, colors))

    color_grid = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3), dtype=np.uint8)
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),(1,1,0),(-1,-1,0),(0,1,1),(0,-1,-1),(1,0,1),(-1,0,-1),(1,-1,0),
                  (-1,1,0),(0,1,-1),(0,-1,1),(1,0,-1),(-1,0,1),(1,1,1),(-1,-1,-1),(1,1,-1),(1,-1,1),(-1,1,1),(-1,-1,1),(-1,1,-1),(1,-1,-1),(0,0,0)]
    for point in normalized_coordinates_with_colors:
        x, y, z, r, g, b = point
        # Arrondir les coordonnées normalisées pour obtenir les indices entiers
        ix, iy, iz = np.round([x, y, z]).astype(int)
        # S'assurer que les indices sont dans les limites de la grille
        if 0 <= ix < grid_size[0] and 0 <= iy < grid_size[1] and 0 <= iz < grid_size[2]:
            # Si la cellule n'a pas encore de couleur (0,0,0), on l'associe à la couleur
            for dx, dy, dz in directions:
                nx, ny, nz = ix + dx, iy + dy, iz + dz
                if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and 0 <= nz < grid_size[2]:
                    if np.all(color_grid[nx, nz, nz] == 0) :
                        color_grid[nx, ny, nz] = [r, g, b]

    
    # Remplir la grille avec une densité ou valeur scalaire
    for x, y, z in normalized_coordinates:
        grid.SetScalarComponentFromDouble(int(x), int(y), int(z), 0, 1.0)

    # Marching Cubes pour l'extraction de l'iso-surface
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(grid)
    marchingCubes.ComputeNormalsOn()
    marchingCubes.SetValue(0, 0.01)  # Valeur pour l'iso-surface
    marchingCubes.Update()

    # Sauvegarder l'iso-surface dans un fichier .vtk
    isoSurface = marchingCubes.GetOutput()
    
    # Récupérer les points de l'iso-surface
    points_iso_surface = isoSurface.GetPoints()
    print(points_iso_surface)
    print("color time")

    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3)  # RGB
    color_array.SetName("Colors")
    # Assigner les couleurs aux points de l'iso-surface
    for i in range(isoSurface.GetNumberOfPoints()):
        point = points_iso_surface.GetPoint(i)
        color=color_grid[int(point[0]),int(point[1]),int(point[2])]
        color_array.InsertNextTuple3(*color)  # RGB, la couleur est un triplet (R, G, B)

    isoSurface.GetPointData().AddArray(color_array)
    
    print(color_array)
    print(isoSurface.GetNumberOfPoints())

    # Sauvegarder l'iso-surface colorée dans un fichier .vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("/home/caro/Documents/Ressources/iso_surface_colored.vtk")  # Nom du fichier de sortie
    writer.SetInputData(isoSurface)
    writer.Write()


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