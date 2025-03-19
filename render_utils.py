import os
import vtk
from vtk.util import numpy_support
import tifffile

def setup_vtk_renderer():
    """
    Sets up a VTK renderer and window.

    Returns:
        (renderer, renderWindow): A tuple containing the set up renderer and render window.
    """
    colors = vtk.vtkNamedColors()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d('LightSlateGray'))
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName('IsoSurface with Marching Cubes')
    renderWindow.AddRenderer(renderer)
    return renderer, renderWindow

def create_sphere_actor(x, y, z, radius, color):
    """
    Creates an actor for a sphere with a given radius and color at a given
    (x, y, z) position.

    Args:
        x (float): The x-coordinate of the sphere's center.
        y (float): The y-coordinate of the sphere's center.
        z (float): The z-coordinate of the sphere's center.
        radius (float): The radius of the sphere.
        color (tuple): A tuple of three floats between 0 and 1 specifying the
            RGB color of the sphere.

    Returns:
        vtkActor: The actor for the sphere.
    """
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(x, y, z)
    sphereSource.SetRadius(radius)
    sphereSource.Update()

    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    sphereActor.GetProperty().SetOpacity(1.0)
    sphereActor.GetProperty().SetColor(color)
    sphereActor.GetProperty().SetInterpolationToFlat()
    
    return sphereActor

def create_tube_actor(x1, y1, z1, x2, y2, z2, radius, color):
    lineSource = vtk.vtkLineSource()
    """
    Creates a tube actor between two points with a specified radius and color.

    Args:
        x1 (float): The x-coordinate of the starting point.
        y1 (float): The y-coordinate of the starting point.
        z1 (float): The z-coordinate of the starting point.
        x2 (float): The x-coordinate of the ending point.
        y2 (float): The y-coordinate of the ending point.
        z2 (float): The z-coordinate of the ending point.
        radius (float): The radius of the tube.
        color (tuple): A tuple of three floats between 0 and 1 specifying the RGB color of the tube.

    Returns:
        vtkActor: The actor representing the tube.
    """

    lineSource.SetPoint1(x1, y1, z1)
    lineSource.SetPoint2(x2, y2, z2)

    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputConnection(lineSource.GetOutputPort())
    tubeFilter.SetRadius(radius)
    tubeFilter.SetNumberOfSides(12)
    tubeFilter.Update()

    tubeMapper = vtk.vtkPolyDataMapper()
    tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())

    tubeActor = vtk.vtkActor()
    tubeActor.SetMapper(tubeMapper)
    tubeActor.GetProperty().SetOpacity(1.0)
    tubeActor.GetProperty().SetColor(color)
    tubeActor.GetProperty().SetInterpolationToFlat()
    
    return tubeActor

class SliceInteractionHandler:
    def __init__(self, renderer, surface_renderer,volume_renderer):
        """
        Initialises the interaction handler with a VTK renderer and a surface renderer.

        Args:
            renderer (vtkRenderer): The VTK renderer.
            surface_renderer (SurfaceRenderer): The surface renderer with the volume data.
            volume_renderer (VolumeRenderer): The volume renderer.
        """
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
        self.volume_renderer=volume_renderer
        self.volume_visibility=True
        self.opacity_factor=volume_renderer.opacity_factor
        

    def on_key_press(self, obj, event):
        """
        Handles key press events for the render window.

        The key press events are as follows:

        - 'd' or 'D': Toggle visibility of all slices.
        - 'k' or 'K': Decrease the offset for the slice index by 1.
        - 'l' or 'L': Increase the offset for the slice index by 1.
        - 'o' or 'O': Toggle visibility of the surface renderer.
        - 'p' or 'P': Toggle visibility of the volume renderer.
        - 'b' or 'B': Decrease the opacity of the volume renderer by 0.5.
        - 'n' or 'N': Increase the opacity of the volume renderer by 0.5.
        - 'Left': Decrease the slice index by the current offset.
        - 'Right': Increase the slice index by the current offset.
        - 'Up': Switch to the next axis (Y-axis -> Z-axis -> X-axis).
        - 'Down': Switch to the previous axis (X-axis -> Y-axis -> Z-axis).
        """
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
        elif key =='p' or key=='P':
            self.toggle_volume()
        elif (key =='b' or key=='B') and self.volume_visibility:
            self.modify_opacity_factor(-0.5)
        elif (key =='n' or key=='N') and self.volume_visibility:
            self.modify_opacity_factor(0.5)
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
        """
        Toggle the visibility of the surface.
        """
        self.surface_renderer.toggle_surface()

    def toggle_volume(self) :
        """
        Toggles the visibility of the volume renderer.

        If the volume is currently visible, it will be hidden.
        If the volume is currently hidden, it will be made visible.
        """

        self.volume_visibility=not self.volume_visibility
        if(self.volume_visibility):
            self.renderer.AddActor(self.volume_renderer.get_volume())
        else:
            self.renderer.RemoveActor(self.volume_renderer.get_volume())

    def modify_opacity_factor(self, factor):
        # Adjust the opacity factor
        """
        Modify the opacity factor of the volume renderer by adding the given factor.

        The opacity factor is a value between 0 and 1 that controls the overall
        transparency of the volume. A value of 1 is completely opaque, while a
        value of 0 is completely transparent.

        The opacity factor is adjusted by adding the given factor to the current
        opacity factor. The volume renderer is then updated to reflect the new
        opacity factor.

        :param factor: The amount to adjust the opacity factor by.
        :type factor: float
        """
        self.opacity_factor += factor
        
        # Update the opacity transfer function of the volume renderer
        self.volume_renderer.opacity_factor = self.opacity_factor
        self.volume_renderer.update_opacity_transfer_function()  # Add a method in VolumeRenderer to update opacity

        # Remove the old volume and add the new one with updated opacity
        self.renderer.RemoveActor(self.volume_renderer.get_volume())  # Remove the previous volume actor
        self.volume_renderer.update_volume()  # Update the volume with new opacity

        # Add the updated volume back to the renderer
        self.renderer.AddActor(self.volume_renderer.get_volume())
        self.renderer.GetRenderWindow().Render()



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
    def __init__(self, tiff_file, x_offset, y_offset, z_offset, pixel_size, offset, decimation_ratio, renderer, surface_file=None):
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
        # Initialize the spatial properties and other parameters
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.pixel_size = pixel_size
        self.decimation_ratio = decimation_ratio
        self.offset = offset
        self.renderer = renderer
        self.tiff_file = tiff_file

        # Load the volumetric data from the TIFF file
        self.volume_data = self.load_tiff(tiff_file)  # Load TIFF data

        # Load a pre-saved surface if available, otherwise, extract a new surface
        if surface_file and os.path.exists(surface_file):
            print(f"Loading pre-saved surface from {surface_file}")
            self.surface = self.load_surface(surface_file)  # Load existing surface
        else:
            # Extract the surface from the volume data
            self.surface = self.extract_surface()  # Extract the surface from the volume

            # Decimate the surface based on the given ratio
            self.decimate_surface(self.decimation_ratio)

            # Save the surface to a file if required
            if surface_file:
                self.save_surface(surface_file)  # Save the surface for future use

        # Create the mapper and actor for the surface
        self.surface_mapper = self.create_surface_mapper()  # Mapper for the surface
        self.surface_actor = self.create_surface_actor()  # Actor to display the surface
        self.slice_actors = self.create_slices_actors()  # Create actors for the slices

    def toggle_surface(self):
        """
        Toggle the visibility of the surface actor in the renderer.
        """
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
        decimate.SetTargetReduction(decimation_ratio)  # Simplification ratio
        # Optionally preserve the topology of the surface
        # decimate.PreserveTopologyOn()  # Preserve surface topology
        decimate.Update()

        # Update the surface with the decimated mesh
        self.surface = decimate.GetOutput()

    def load_tiff(self, tiff_file):
        """
        Loads a TIFF file with tifffile and returns a vtkImageData object.
        
        Args:
            tiff_file (str): Path to the TIFF file containing volumetric data.
        
        Returns:
            vtk.vtkImageData: The loaded volumetric data as vtkImageData.
        """
        volume_data = tifffile.imread(tiff_file)

        z_size, y_size, x_size = volume_data.shape

        # Convert the numpy array to vtkImageData
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
        
        Returns:
            vtkPolyData: The extracted surface from the volume data.
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
        
        Returns:
            vtk.vtkPolyDataMapper: The mapper for the surface data.
        """
        surface_mapper = vtk.vtkPolyDataMapper()
        surface_mapper.SetInputData(self.surface)  # Surface extracted by MarchingCubes
        surface_mapper.SetScalarVisibility(True)

        # Create a lookup table for grayscale rendering
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0, 0)  # Use grayscale
        lut.SetSaturationRange(0, 0)  # No saturation
        lut.SetValueRange(0, 1)  # Default range

        # Apply the lookup table to the surface mapper
        surface_mapper.SetLookupTable(lut)
        return surface_mapper

    def create_surface_actor(self):
        """
        Creates an actor to render the surface.
        
        Returns:
            vtk.vtkActor: The actor for rendering the surface.
        """
        surface_actor = vtk.vtkActor()
        surface_actor.SetMapper(self.surface_mapper)
        surface_actor.GetProperty().SetColor(1, 0, 0)  # Example color: red
        surface_actor.GetProperty().SetOpacity(0.2)
        return surface_actor

    def create_slices_actors(self):
        """
        Creates actors for each slice in the volume for each axis.

        Returns a list of lists of actors, where the outer list contains one
        inner list for each axis (X, Y, Z), and the inner list contains one
        actor for each slice in the volume along that axis.

        The opacity of each actor is set to 0.6.
        
        Returns:
            list: A list of slice actors for each axis (X, Y, Z).
        """
        actors = []

        x_size, y_size, z_size = self.volume_data.GetDimensions()

        # Loop through each axis (X, Y, Z) to create slice actors
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
        
        This method initializes a render window and interactor, adds the surface actor,
        and starts the rendering loop.
        """
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window_interactor = vtk.vtkRenderWindowInteractor()

        render_window.AddRenderer(renderer)
        render_window_interactor.SetRenderWindow(render_window)

        # Add the surface actor to the renderer
        renderer.AddActor(self.surface_actor)

        # Initialize and start the render window interaction
        render_window_interactor.Initialize()
        render_window.Render()
        render_window.SetWindowName("Surface Rendering - Marching Cubes")
        render_window_interactor.Start()


class VolumeRenderer:
    def __init__(self, tiff_file, x_offset, y_offset, z_offset, pixel_size, opacity_factor):
        """
        Initialises the VolumeRenderer with the given TIFF file and spatial information.
        
        Args:
            tiff_file (str): Path to the TIFF file containing volumetric data.
            x_offset (float): The x-axis offset for the origin of the volume.
            y_offset (float): The y-axis offset for the origin of the volume.
            z_offset (float): The z-axis offset for the origin of the volume.
            pixel_size (float): The size of each voxel in the volume.
            opacity_factor (float): The opacity factor for the volume rendering.
        """
        self.tiff_file = tiff_file
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.pixel_size = pixel_size
        self.opacity_factor = opacity_factor

        # Initialisation du lecteur TIFF
        self.reader = vtk.vtkTIFFReader()
        self.reader.SetFileName(tiff_file)
        self.reader.SetOrientationType(1)  # Adapter selon l'orientation de votre image
        self.reader.Update()

        self.image_data = self.reader.GetOutput()
        self.image_data.SetSpacing(pixel_size, pixel_size, pixel_size)

        # Configuration du mapper et du volume
        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        self.volume_mapper.SetInputConnection(self.reader.GetOutputPort())

        self.color_transfer_function = self.create_color_transfer_function()
        self.opacity_transfer_function = self.create_opacity_transfer_function()

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.color_transfer_function)
        self.volume_property.SetScalarOpacity(self.opacity_transfer_function)
        self.volume_property.ShadeOn()  # Activer les ombres pour un rendu réaliste
        self.volume_property.SetInterpolationTypeToLinear()

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        self.volume.SetPosition(x_offset * pixel_size, y_offset * pixel_size, z_offset * pixel_size)

    def create_color_transfer_function(self):
        """
        Creates a color transfer function to map scalar values to RGB colors.

        This function defines a series of color points at specific scalar values,
        ranging from black to white with various color transitions, including shades
        of purple, pink, red, orange, yellow, and white.

        Returns:
            vtk.vtkColorTransferFunction: A color transfer function with predefined RGB values.
        """
        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(30, 0.0, 0.0, 0.0)  # Black
        color_transfer_function.AddRGBPoint(55, 0.1, 0.0, 0.5)  # Dark bluish violet
        color_transfer_function.AddRGBPoint(60, 0.4, 0.0, 0.4)  # Dark violet
        color_transfer_function.AddRGBPoint(65, 0.6, 0.0, 0.6)  # Violet pink
        color_transfer_function.AddRGBPoint(70, 0.8, 0.3, 0.4)  # Light pink to red
        color_transfer_function.AddRGBPoint(75, 0.9, 0.5, 0.1)  # Red to orange
        color_transfer_function.AddRGBPoint(80, 1.0, 0.5, 0.0)  # Orange
        color_transfer_function.AddRGBPoint(90, 1.0, 0.7, 0.0)  # Orange to yellow
        color_transfer_function.AddRGBPoint(100, 1.0, 0.9, 0.2) # Yellow
        color_transfer_function.AddRGBPoint(130, 1.0, 1.0, 1.0) # White

        return color_transfer_function


    def create_opacity_transfer_function(self):
        """
        Creates an opacity transfer function to map scalar values to opacity.

        This function defines the opacity values at different scalar points,
        ranging from fully transparent to nearly opaque. The opacity increases
        with higher scalar values, with the factor of opacity being influenced
        by the `opacity_factor` parameter.

        Returns:
            vtk.vtkPiecewiseFunction: An opacity transfer function with predefined opacity values.
        """
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)  # Fully transparent
        opacity_transfer_function.AddPoint(10, 0.0)  # Fully transparent
        opacity_transfer_function.AddPoint(30, 0.0)  # Semi-transparent
        opacity_transfer_function.AddPoint(50, 0.005 * self.opacity_factor)
        opacity_transfer_function.AddPoint(60, 0.01 * self.opacity_factor)
        opacity_transfer_function.AddPoint(100, 0.015 * self.opacity_factor)
        opacity_transfer_function.AddPoint(130, 0.02 * self.opacity_factor)  # Almost opaque

        return opacity_transfer_function


    def get_volume(self):
        """
        Returns the vtkVolume object for rendering.

        This function retrieves the `vtkVolume` object, which contains the volume data
        and is ready to be added to a VTK renderer for visualization.

        Returns:
            vtk.vtkVolume: The vtkVolume object to be rendered.
        """
        return self.volume


    def update_opacity_transfer_function(self):
        """ Update the opacity transfer function with the new opacity factor. """
        self.opacity_transfer_function = self.create_opacity_transfer_function()  # Recreate the opacity transfer function
        self.volume_property.SetScalarOpacity(self.opacity_transfer_function)  # Update the volume's opacity

    def update_volume(self):
        """ Update the volume actor with the new opacity settings. """
        # You can call update or refresh the volume here.
        self.volume.SetProperty(self.volume_property) 