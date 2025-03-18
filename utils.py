import csv
import numpy as np
from scipy.interpolate import splev, splprep,splrep
import vtk
from vedo import utils
import tifffile
from vtk.util import numpy_support
import os
import cv2
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera


def get_curve(csv_file, spacing=10):
    """
    Reads a CSV file and returns a resampled list of 3D points that form a curve.
    """
    curve_points = []
    with open(csv_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader)
        for row in reader:
            curve_points.append([float(row[0]), float(row[1]), float(row[2])])
    
    return resample_curve(curve_points, spacing)


def resample_curve(curve, spacing):
    """
    Resample a curve of 3D points at evenly spaced intervals.
    """
    curve = np.array(curve)
    curve_length = np.sum(np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1)))
    num_points = int(np.ceil(curve_length / spacing))

    tck, u = splprep(curve.T, s=0, per=False)
    u_new = np.linspace(0, 1, num_points)
    return np.column_stack(splev(u_new, tck, der=0)).tolist()


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
        (0.0, 0.0, 1.0),   # Bleu
        (1.0, 0.0, 0.0),   # Rouge
        (0.0, 0.5, 0.0),   # Vert
        (1.0, 0.5, 0.0),   # Orange
        (1.0, 1.0, 0.0),   # Jaune
        (0.0, 1.0, 1.0),   # Turquoise
        (0.5, 0.25, 0.0),  # Marron caramel
        (0.0, 0.0, 0.0),   # Gris noir
    ]

    palettes = []
    for i in range(n_dir):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()

        base_color = base_colors[i % len(base_colors)]

        for j in range(256):
            intensity = j / 255.0  # De foncé (0) à clair (1)
            r = base_color[0] * intensity
            g = base_color[1] * intensity
            b = base_color[2] * intensity
            lut.SetTableValue(j, r, g, b, 1.0)

        palettes.append(lut)
    return palettes


def get_color_from_z(z, z_min, z_max, dir_index, palettes):

    """
    Retrieves the RGB color for a given z-coordinate from a specified color palette.

    This function normalizes the input z value to a range between 0 and 1, ensuring
    it is within bounds. It then uses the direction index to select the appropriate
    color palette and retrieves the corresponding color based on the normalized z value.

    Args:
        z (float): The z-coordinate value for which the color is to be determined.
        z_min (float): The minimum z-coordinate value in the dataset.
        z_max (float): The maximum z-coordinate value in the dataset.
        dir_index (int): The index of the direction which determines the color palette.
        palettes (list): A list of vtkLookupTable objects representing color palettes.

    Returns:
        tuple: A tuple containing the RGB values for the specified z-coordinate.
    """

    normalized_z = 1 - (z - z_min) / (z_max - z_min)
    normalized_z = np.clip(normalized_z, 0, 1) 

    lookup_table = palettes[dir_index]
    lookup_table.SetTableRange(0.0, 1.0)
    
    color = lookup_table.GetTableValue(int(normalized_z * 255))[:3]
    return color


def get_information(information_path):
    """
    Reads a CSV file containing curve and radius information and returns it as a list of rows.

    Args:
        information_path (str): The path to the CSV file containing the information.

    Returns:
        list: A list of rows, where each row is a list of strings representing the curve and radius information.
    """
    information=[]
    with open(information_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader)
        for row in reader :
            information.append(row)
    return information


def get_radius(radius_path, target_lengths):
    """
    Reads a CSV file containing radius values and resamples them to match target lengths.
    """
    radius = []
    
    with open(radius_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        slice_values = next(reader)  # Read the header row
        # Convert header values to floats if possible
        slice_values = [float(val) if is_float(val) else val for val in slice_values]
        
        # Collect radius data in a single list, avoiding repeated appends
        for row in reader:
            radius.extend([[slice_values[i], float(value)] for i, value in enumerate(row)])
    
    # Split the radius into chunks based on the number of slice values
    chunk_size = len(slice_values)
    radius_split = [radius[i:i + chunk_size] for i in range(0, len(radius), chunk_size)]
    
    # Resample and return
    return resample_radius(radius_split, target_lengths)

def is_float(value):
    """
    Helper function to check if a string can be converted to a float.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def resample_radius(radius_split, target_lengths):
    """
    Resamples a list of radius data chunks to match target lengths.
    """
    resampled_chunks = []
    
    # Iterate over the chunks and target lengths
    for chunk, num_points in zip(radius_split, target_lengths):
        chunk = np.array(chunk)
        slice_values = chunk[:, 0]
        values = chunk[:, 1]

        # Resample using splev and splrep
        x = np.arange(len(slice_values))
        x_new = np.linspace(0, len(slice_values) - 1, num_points)

        slice_resampled = splev(x_new, splrep(x, slice_values))
        values_resampled = splev(x_new, splrep(x, values))

        # Stack the resampled values and append to the result list
        radius_resampled = np.column_stack((slice_resampled, values_resampled))

        # Ensure no negative radius values by applying np.maximum
        radius_resampled[:, 1] = np.maximum(radius_resampled[:, 1], 0)

        resampled_chunks.append(radius_resampled.tolist())

    return resampled_chunks

def create_direction_cosines_from_point_list(polyline):
    """
    Computes the direction cosines from a list of points forming a polyline.

    This function calculates the normal vectors between consecutive points in
    the polyline and derives the orthogonal direction cosines using cross
    products. It returns three lists representing the three orthogonal axes
    (axiss0, axiss1, axiss2) for each segment between the points in the
    polyline.

    Args:
        polyline (list of list of floats): A list of 3D points defining the polyline.

    Returns:
        tuple: Three lists of 3D vectors representing the orthogonal axes
               (axiss0, axiss1, axiss2) for each segment in the polyline.
    """

    normals, axiss0, axiss1, axiss2 = [], [], [], []
    N = len(polyline)
    for i in range(N - 1):
        p0, p1 = polyline[i], polyline[i + 1]
        normals.append([-p1[0] + p0[0], -p1[1] + p0[1], -p1[2] + p0[2]])

    normal = normals[0]
    axis2 = utils.versor(normal)
    axisTemp = [1, 0, 0] if np.abs(normal[0]) < min(np.abs(normal[1]), np.abs(normal[2])) \
        else [0, 1, 0] if np.abs(normal[1]) < np.abs(normal[2]) else [0, 0, 1]
    axis0 = utils.versor(np.cross(axisTemp, axis2))
    axis1 = utils.versor(np.cross(axis2, axis0))
    axiss0.append(axis0)
    axiss1.append(axis1)
    axiss2.append(axis2)

    for i in range(1, N - 1):
        normal = normals[i]
        axis2 = utils.versor(normal)
        axis0Prev = axiss0[i - 1]
        axis1 = utils.versor(np.cross(axis2, axis0Prev))
        axis0 = utils.versor(np.cross(axis1, axis2))
        axiss0.append(axis0)
        axiss1.append(axis1)
        axiss2.append(axis2)

    return axiss0, axiss1, axiss2



class SliceInteractionHandler:
    def __init__(self, renderer, surface_renderer):
        self.renderer = renderer
        self.surface_renderer = surface_renderer
        self.current_xslice_index = int(self.surface_renderer.volume_data.GetDimensions()[0]/2 )# Index of the current slice
        self.current_yslice_index = int(self.surface_renderer.volume_data.GetDimensions()[1]/2)
        self.current_zslice_index = int(self.surface_renderer.volume_data.GetDimensions()[2]/2)
        self.current_axis = 2  # Default to Z-axis
        self.previous_xslice_index=0
        self.previous_yslice_index=0
        self.previous_zslice_index=0
        self.previous_axis=2
        self.slices_visible_x = True
        self.slices_visible_y = True
        self.slices_visible_z = True
        self.offset=self.surface_renderer.offset
        self.min_axis_size=min(self.surface_renderer.volume_data.GetDimensions())
        

    def on_key_press(self, obj, event):
        key = obj.GetKeySym()  # Get the key that was pressed
        if key == 'd' or key == 'D':
            # Toggle visibility of all slices with the 'D' key
            self.toggle_slices_visibility()
        elif key =='k' or key=='K':
            self.offset=max(0,self.offset-1)
        elif key == 'l' or key=='L':
            self.offset = min(self.offset + 1, self.min_axis_size - 1)
        elif key == 'o' or key=='O':
            self.toggle_surface()
        elif key == 'Left':
        # Decrease slice index
            self.update_slice(-self.offset)
        elif key == 'Right':
            # Increase slice index
            self.update_slice(self.offset)
        elif key == 'Up':
            # Switch to the next axis (Y-axis -> Z-axis -> X-axis)
            self.previous_axis=self.current_axis
            self.current_axis = (self.current_axis + 1) % 3
            self.current_slice_index = 0  # Reset to the first slice of the new axis
            self.update_renderer_with_new_slice()
        elif key == 'Down':
            # Switch to the previous axis (X-axis -> Y-axis -> Z-axis)
            self.previous_axis=self.current_axis
            self.current_axis = (self.current_axis - 1) % 3
            self.current_slice_index = 0  # Reset to the first slice of the new axis
            self.update_renderer_with_new_slice()
 
            
    def toggle_surface(self) :
        self.surface_renderer.toggle_surface()

    def update_slice(self, direction):
        """
        Update the current slice along the Z-axis or other axes.
        """
        # Get the number of slices along the current axis
        axis_size = self.surface_renderer.volume_data.GetDimensions()[self.current_axis]

        # Calculate new slice index
        if direction < 0:
            if self.current_axis == 0 and self.slices_visible_x:
                new_index = max(0, self.current_xslice_index - self.offset)
            elif self.current_axis == 1 and self.slices_visible_y:
                new_index = max(0, self.current_yslice_index - self.offset)
            elif self.current_axis == 2 and self.slices_visible_z:
                new_index = max(0, self.current_zslice_index - self.offset)
        elif direction > 0 : 
            if self.current_axis == 0 and self.slices_visible_x:
                new_index = min(axis_size - 1, self.current_xslice_index + self.offset)
            elif self.current_axis == 1 and self.slices_visible_y:
                new_index = min(axis_size - 1, self.current_yslice_index + self.offset)
            elif self.current_axis == 2 and self.slices_visible_z:   
                new_index = min(axis_size - 1, self.current_zslice_index + self.offset)   
        if self.current_axis == 0 and self.slices_visible_x:
            self.previous_xslice_index = self.current_xslice_index
            self.current_xslice_index = new_index
            self.previous_axis=self.current_axis
            self.update_renderer_with_new_slice()
        elif self.current_axis == 1 and self.slices_visible_y:
            self.previous_yslice_index = self.current_yslice_index
            self.current_yslice_index = new_index
            self.previous_axis=self.current_axis
            self.update_renderer_with_new_slice()
        elif self.current_axis == 2 and self.slices_visible_z:
            self.previous_zslice_index = self.current_zslice_index
            self.current_zslice_index = new_index
            self.previous_axis=self.current_axis
            self.update_renderer_with_new_slice()
        
        

    def update_renderer_with_new_slice(self):
        """
        Remove the previous slice and add the new slice to the renderer.
        """
        # Remove the current slice actor from the renderer

        self.remove_current_slice_actor()
        # Add the new slice actor for the current axis
        if(self.current_axis==0 and self.slices_visible_x):
            self.renderer.AddActor(self.surface_renderer.slice_actors[self.current_axis][self.current_xslice_index])
        elif(self.current_axis==1 and self.slices_visible_y):
            self.renderer.AddActor(self.surface_renderer.slice_actors[self.current_axis][self.current_yslice_index])
        elif(self.current_axis==2 and self.slices_visible_z):
            self.renderer.AddActor(self.surface_renderer.slice_actors[self.current_axis][self.current_zslice_index])

        # Update the render window
        self.renderer.GetRenderWindow().Render()

    def remove_current_slice_actor(self):
        """
        Remove the current slice actor for the current axis.
        """
        # Ensure we are removing the correct slice actor
        # Remove only the specific slice actor at the current index
        if(self.previous_axis==0):
            self.renderer.RemoveActor(self.surface_renderer.slice_actors[self.previous_axis][self.previous_xslice_index])
        elif(self.previous_axis==1):
            self.renderer.RemoveActor(self.surface_renderer.slice_actors[self.previous_axis][self.previous_yslice_index])
        else:
            self.renderer.RemoveActor(self.surface_renderer.slice_actors[self.previous_axis][self.previous_zslice_index])


    def toggle_slices_visibility(self):
        """
        Toggle visibility of all slices (X, Y, Z).
        """
        if(self.current_axis==0):
            if self.slices_visible_x:
                self.renderer.RemoveActor(self.surface_renderer.slice_actors[self.current_axis][self.current_xslice_index])
            else : 
                self.renderer.AddActor(self.surface_renderer.slice_actors[self.current_axis][self.current_xslice_index])
            self.slices_visible_x = not self.slices_visible_x
        elif(self.current_axis==1) :
            if self.slices_visible_y:
                self.renderer.RemoveActor(self.surface_renderer.slice_actors[self.current_axis][self.current_yslice_index])
            else : 
                self.renderer.AddActor(self.surface_renderer.slice_actors[self.current_axis][self.current_yslice_index])
            self.slices_visible_y = not self.slices_visible_y
        else : 
            if self.slices_visible_z:
                self.renderer.RemoveActor(self.surface_renderer.slice_actors[self.current_axis][self.current_zslice_index])
            else : 
                self.renderer.AddActor(self.surface_renderer.slice_actors[self.current_axis][self.current_zslice_index])
            self.slices_visible_z = not self.slices_visible_z

        # Update the render window
        self.renderer.GetRenderWindow().Render()



class SurfaceRenderer:
    def __init__(self, tiff_file, x_offset, y_offset, z_offset, pixel_size, offset, decimation_ratio, renderer,surface_file=None):
        """
        Initializes the SurfaceRenderer with the provided TIFF file and spatial information.
        Optionally loads a pre-saved surface from a file.
        
        Args:
            tiff_file (str): Path to the TIFF file containing volumetric data.
            x_offset (float): The x-axis offset for the origin of the volume.
            y_offset (float): The y-axis offset for the origin of the volume.
            z_offset (float): The z-axis offset for the origin of the volume.
            pixel_size (float): The size of each voxel in the volume.
            decimation_ratio (float): The ratio of surface simplification (0.0 to 1.0).
            surface_file (str): Optional path to a pre-saved surface file (.vtk or .stl).
        """
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.pixel_size = pixel_size
        self.decimation_ratio = decimation_ratio
        self.offset = offset
        self.renderer=renderer

        # Charger les données volumétriques depuis le fichier TIFF
        self.volume_data = self.load_tiff(tiff_file)  # Charger les données TIFF

        if surface_file and os.path.exists(surface_file):
            print(f"Loading pre-saved surface from {surface_file}")
            self.surface = self.load_surface(surface_file)  # Charger la surface existante
        else:
            # Extraire la surface depuis les données volumétriques
            self.surface = self.extract_surface()  # Extraire la surface depuis le volume

            # Décimer la surface
            self.decimate_surface(self.decimation_ratio)

            # Sauvegarder la surface pour une utilisation future
            if surface_file:
                self.save_surface(surface_file)  # Sauvegarder la surface dans le fichier spécifié

        # Créer le mapper et l'acteur pour la surface
        self.surface_mapper = self.create_surface_mapper()  # Mapper pour la surface
        self.surface_actor = self.create_surface_actor()  # Acteur pour afficher la surface
        self.slice_actors = self.create_slices_actors()  # Créer les acteurs pour les slices


    def toggle_surface(self) :
        self.surface_actor.SetVisibility(not self.surface_actor.GetVisibility())
        self.renderer.GetRenderWindow().Render()
    def save_surface(self, filename):
        """
        Saves the surface as a VTK file.
        
        Args:
            filename (str): The path where to save the surface file.
        """
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(self.surface)
        writer.Write()
        print(f"Surface saved to {filename}")

    def load_surface(self, filename):
        """
        Loads a surface from a VTK file.
        
        Args:
            filename (str): The path to the surface file.
        
        Returns:
            vtkPolyData: The loaded surface.
        """
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        return reader.GetOutput()

    def decimate_surface(self, decimation_ratio):
        """
        Simplifies the surface using vtkDecimatePro or vtkQuadricDecimation.

        Args:
            decimation_ratio (float): The ratio of simplification (0.0 to 1.0).
                                      1.0 means no simplification, 0.0 means complete reduction.
        """
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(self.surface)
        decimate.SetTargetReduction(decimation_ratio)  # Ratio de simplification
        #decimate.PreserveTopologyOn()  # Preserving topology of the surface
        decimate.Update()

        # Mise à jour de la surface avec le maillage simplifié
        self.surface = decimate.GetOutput()

    def load_tiff(self, tiff_file):
        """
        Loads a TIFF file with tifffile and returns a vtkImageData object.
        """
        volume_data = tifffile.imread(tiff_file)

        z_size, y_size, x_size = volume_data.shape

        # Convert to vtkImageData
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(x_size, y_size, z_size)
        image_data.SetSpacing(self.pixel_size, self.pixel_size, self.pixel_size)
        image_data.SetOrigin(self.x_offset * self.pixel_size,
                             self.y_offset * self.pixel_size,
                             self.z_offset * self.pixel_size)

        vtk_array = vtk.util.numpy_support.numpy_to_vtk(volume_data.flatten(), deep=True)
        image_data.GetPointData().SetScalars(vtk_array)

        return image_data

    def extract_surface(self):
        """
        Uses vtkMarchingCubes to extract the surface of the volume.
        """
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(self.volume_data)  # Volumetric data
        marching_cubes.ComputeNormalsOn()
        marching_cubes.SetValue(0, 100)  # Isosurface value to extract the surface
        marching_cubes.Update()
        return marching_cubes.GetOutput()

    def create_surface_mapper(self):
        """
        Creates a mapper for the extracted surface.
        """
        surface_mapper = vtk.vtkPolyDataMapper()
        surface_mapper.SetInputData(self.surface)  # Surface extracted by MarchingCubes
        surface_mapper.SetScalarVisibility(False)
        return surface_mapper

    def create_surface_actor(self):
        """
        Creates an actor to render the surface.
        """
        surface_actor = vtk.vtkActor()
        surface_actor.SetMapper(self.surface_mapper)
        surface_actor.GetProperty().SetColor(1, 0, 0)  # Example color: red
        surface_actor.GetProperty().SetOpacity(0.2)
        return surface_actor

    def create_slices_actors(self):
        """ Crée les acteurs pour afficher toutes les slices sur chaque axe. """
        actors = []

        x_size, y_size, z_size = self.volume_data.GetDimensions()

        for axis, orientation in zip(["X", "Y", "Z"], [0, 1, 2]):
            slice_actors_for_axis = []
            for i in range(self.volume_data.GetDimensions()[orientation]):
                mapper = vtk.vtkImageSliceMapper()
                mapper.SetInputData(self.volume_data)
                mapper.SetOrientation(orientation)
                mapper.SetSliceNumber(i)

                actor = vtk.vtkImageActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetOpacity(0.6)

                slice_actors_for_axis.append(actor)

            actors.append(slice_actors_for_axis)

        return actors

    def render(self):
        """
        Renders the surface with a surface rendering type.
        """
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window_interactor = vtk.vtkRenderWindowInteractor()

        render_window.AddRenderer(renderer)
        render_window_interactor.SetRenderWindow(render_window)

        # Add the surface actor to the scene
        renderer.AddActor(self.surface_actor)

        # Initialize and display the render window
        render_window_interactor.Initialize()
        render_window.Render()
        render_window.SetWindowName("Surface Rendering - Marching Cubes")
        render_window_interactor.Start()


def remplacer_pixels_dossier(dossier_entree, dossier_sortie,value):
    
    """
    Remplace les pixels d'une image qui valent x par y.

    Args:
        dossier_entree (str): Dossier contenant les images à modifier.
        dossier_sortie (str): Dossier où seront sauvegardées les images modifiées.

    Returns:
        None
    """
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
    
    # Liste des fichiers dans le dossier d'entrée
    images = [f for f in os.listdir(dossier_entree) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Traiter chaque image dans le dossier
    for img_nom in images:
        image_path = os.path.join(dossier_entree, img_nom)
        output_path = os.path.join(dossier_sortie, img_nom)

        # Charger l'image en niveaux de gris
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
        
        # Vérifier si l'image est bien chargée
        if image is None:
            print(f"Erreur lors du chargement de l'image {image_path}")
            continue
        
        # Remplacer les pixels qui valent x par y
        image[image == value] = 0
        image[image>value] = image[image>value]-1
        
        # Sauvegarder l'image modifiée
        cv2.imwrite(output_path, image)
        print(f"Image modifiée sauvegardée : {output_path}")


def remplacer_pixels(image_path, x, y, output_path):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
    
    # Vérifier si l'image est bien chargée
    if image is None:
        print(f"Erreur lors du chargement de l'image {image_path}")
        return
    
    print(f"Type de l'image : {type(image)}")
    print(f"Shape de l'image : {image.shape if image is not None else 'None'}")

    # Remplacer les pixels qui valent x par y
    image[image == x] = y
    print(np.unique(image))

    # Sauvegarder l'image modifiée
    cv2.imwrite(output_path, image)
    print(f"Image modifiée sauvegardée : {output_path}")
