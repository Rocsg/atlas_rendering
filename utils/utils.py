import vtk
import numpy as np

def create_color_palettes(n_dir):
    """
    Creates a list of color palettes based on a set of base colors.

    Each color palette is a vtkLookupTable that maps intensity values (0 to 1)
    to colors. The colors range from dark (0) to bright (1) for each base color.

    Args:
        n_dir (int): The number of color palettes to generate.

    Returns:
        list: A list of vtkLookupTable objects, each representing a color palette.
    """
    base_colors = [
        (0.0, 0.0, 1.0),   # Blue
        (1.0, 0.0, 0.0),   # Red
        (0.0, 0.5, 0.0),   # Green
        (1.0, 0.5, 0.0),   # Orange
        (1.0, 1.0, 0.0),   # Yellow
        (0.0, 1.0, 1.0),   # Turquoise
        (0.5, 0.25, 0.0),  # Caramel Brown
        (0.5, 0.0, 0.5),   # Purple
    ]

    palettes = []
    for i in range(n_dir):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()

        base_color = base_colors[i % len(base_colors)]

        for j in range(256):
            intensity = j / 255.0  # From dark (0) to bright (1)
            r = base_color[0] * intensity
            g = base_color[1] * intensity
            b = base_color[2] * intensity
            lut.SetTableValue(j, r, g, b, 1.0)

        palettes.append(lut)
    return palettes


def count_vertices_and_triangles(renderer):
    """
    Counts the total number of vertices and triangles in the scene.
    
    Iterates over all actors in the renderer and counts the vertices and triangles 
    of their associated polydata.

    Args:
        renderer (vtkRenderer): The renderer containing the actors to be counted.

    Returns:
        tuple: A tuple containing the total number of vertices and triangles.
    """
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
    Chooses a color from the color palette based on the z-coordinate and direction index.
    
    Normalizes the z value and retrieves the corresponding color from the appropriate
    color palette (based on the direction index).

    Args:
        z (float): The z-coordinate to be colored.
        z_min (float): The minimum value of z in the dataset.
        z_max (float): The maximum value of z in the dataset.
        dir_index (int): The direction index to choose the color palette.
        palettes (list): A list of vtkLookupTable objects representing color palettes.

    Returns:
        tuple: A tuple (r, g, b) representing the color for the given z-coordinate.
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
    
    Normalizes the z value between 0 and 1, and then applies a predefined "jet"
    color palette that ranges from blue (low values) to red (high values).

    Args:
        z (float): The z-coordinate to be colored.
        z_min (float): The minimum value of z in the dataset.
        z_max (float): The maximum value of z in the dataset.

    Returns:
        tuple: A tuple (r, g, b) representing the color for the given z-coordinate.
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
    """
    Normalizes a vector (makes its length 1).
    
    Args:
        v (np.array): The vector to normalize.
    
    Returns:
        np.array: The normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Function to calculate the rotation angle and axis based on the direction vector
def calculate_rotation_vector(v):
    """
    Calculates the rotation axis and angle to align a vector with the reference axis (Z-axis).

    Args:
        v (np.array): The direction vector to rotate.

    Returns:
        tuple: A tuple (axis, angle) representing the rotation axis and angle.
    """
    # Direction vector of the cylinder
    direction = np.array(v)
    
    # Reference vector (assuming the Z-axis is the reference for rotation)
    reference = np.array([0.0, 0.0, 1.0])

    # Dot product to calculate the angle
    dot_product = np.dot(reference, direction)
    angle = np.arccos(dot_product)

    # Cross product to calculate the rotation axis
    axis = np.cross(reference, direction)
    
    # Normalize the rotation axis
    axis = normalize(axis)
    
    return axis, np.degrees(angle)

def compute_z_min_max(z_values):
    """
    Computes the minimum and maximum values of a list of z-coordinates.

    Args:
    - z_values (list): List of z-coordinates.

    Returns:
    - tuple: A tuple (min_z, max_z) containing the minimum and maximum z values.
    """
    return min(z_values), max(z_values)

def check_memory_usage():
    """
    Affiche l'utilisation de la mémoire du système.
    """
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    memory_usage = mem_info.rss / 1024 / 1024  # Converti la mémoire en Mo
    print(f"Utilisation de la mémoire : {memory_usage:.2f} Mo")

