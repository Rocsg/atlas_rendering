import os
from types import SimpleNamespace
import numpy as np
import json
import vtk
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
from utils.data_utils import get_sorted_vtk_files
from utils.utils import  count_vertices_and_triangles
from utils.render_utils import SurfaceRenderer, SliceInteractionHandler, VolumeRenderer,setup_vtk_renderer


def load_config(config_path="config.json"):
    """Loads the configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def setup_interaction(renderer, surface_renderer, volume_renderer):
    # Create an event handler for the arrow keys
    interaction_handler = SliceInteractionHandler(renderer, surface_renderer, volume_renderer)
    
    # Create an interactor and attach the event handler
    renderWindowInteractor = vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleUnicam()
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindowInteractor.SetRenderWindow(renderer.GetRenderWindow())
    interaction_handler.create_sliders()
    interaction_handler.create_text()
    interaction_handler.create_button()
    renderWindowInteractor.AddObserver('KeyPressEvent', interaction_handler.on_key_press)

    return renderWindowInteractor

def render_surfaces(renderer,leaf_directory,root_directory,inner_opacity,outer_opacity) :
    root_files_in = get_sorted_vtk_files(os.path.join(root_directory, "stem_reduced"))
    root_files_out = get_sorted_vtk_files(os.path.join(root_directory, "cylinder_reduced"))
    leaf_files_in = get_sorted_vtk_files(os.path.join(leaf_directory, "stem_reduced"))
    leaf_files_out = get_sorted_vtk_files(os.path.join(leaf_directory, "cylinder_reduced"))
    
    num_files_root = min(len(root_files_in), len(root_files_out))
    if num_files_root == 0:
        print("Aucun fichier .vtk trouvé dans les dossiers spécifiés.")
        return

    num_files_leaf = min(len(leaf_files_in), len(leaf_files_out))
    if num_files_leaf == 0:
        print("Aucun fichier .vtk trouvé dans les dossiers spécifiés.")
        return
    
    # Processing roots
    for i in range(num_files_root):
        reader_root = vtk.vtkPolyDataReader()
        reader_root.SetFileName(root_files_out[i])
        reader_root.Update()
        polydata_root = reader_root.GetOutput()
        
        reader_stem = vtk.vtkPolyDataReader()
        reader_stem.SetFileName(root_files_in[i])
        reader_stem.Update()
        polydata_stem = reader_stem.GetOutput()
        
        # Color for cylinders of roots
        colors_root = polydata_root.GetPointData().GetArray("Colors")
        color_data_root = vtk.vtkUnsignedCharArray()
        color_data_root.SetName("Colors")
        color_data_root.SetNumberOfComponents(3)
        
        if colors_root:
            for j in range(colors_root.GetNumberOfTuples()):
                r, g, b = colors_root.GetTuple(j)
                color_data_root.InsertNextTuple3(int(r * 255), int(g * 255), int(b * 255))
        else:
            for j in range(polydata_root.GetNumberOfPoints()):
                color_data_root.InsertNextTuple3(25, 25, 25)
        
        polydata_root.GetPointData().SetScalars(color_data_root)
        
        # Color for stems of roots
        colors_stem = polydata_stem.GetPointData().GetArray("Colors")
        color_data_stem = vtk.vtkUnsignedCharArray()
        color_data_stem.SetName("Colors")
        color_data_stem.SetNumberOfComponents(3)
        
        if colors_stem:
            for j in range(colors_stem.GetNumberOfTuples()):
                r, g, b = colors_stem.GetTuple(j)
                color_data_stem.InsertNextTuple3(r,g,b)
        
        polydata_stem.GetPointData().SetScalars(color_data_stem)
        
        # Mappers
        mapper_root = vtk.vtkPolyDataMapper()
        mapper_root.SetInputData(polydata_root)
        
        mapper_stem = vtk.vtkPolyDataMapper()
        mapper_stem.SetInputData(polydata_stem)
        
        # Actors
        actor_root = vtk.vtkActor()
        actor_root.SetMapper(mapper_root)
        actor_root.GetProperty().SetOpacity(outer_opacity)
        
        actor_stem = vtk.vtkActor()
        actor_stem.SetMapper(mapper_stem)
        actor_stem.GetProperty().SetOpacity(inner_opacity)
        
        # Assembly
        assembly = vtk.vtkAssembly()
        assembly.AddPart(actor_root)
        assembly.AddPart(actor_stem)
        
        renderer.AddActor(assembly)

    #Processing leaves
    for i in range(num_files_leaf):
        reader_root = vtk.vtkPolyDataReader()
        reader_root.SetFileName(leaf_files_out[i])
        reader_root.Update()
        polydata_root = reader_root.GetOutput()
        
        reader_stem = vtk.vtkPolyDataReader()
        reader_stem.SetFileName(leaf_files_in[i])
        reader_stem.Update()
        polydata_stem = reader_stem.GetOutput()
        
        # Color for cylinders of leaves
        colors_root = polydata_root.GetPointData().GetArray("Colors")
        color_data_root = vtk.vtkUnsignedCharArray()
        color_data_root.SetName("Colors")
        color_data_root.SetNumberOfComponents(3)
        
        if colors_root:
            for j in range(colors_root.GetNumberOfTuples()):
                r, g, b = colors_root.GetTuple(j)
                color_data_root.InsertNextTuple3(int(r * 255), int(g * 255), int(b * 255))
        else:
            for j in range(polydata_root.GetNumberOfPoints()):
                color_data_root.InsertNextTuple3(25, 25, 25)
        
        polydata_root.GetPointData().SetScalars(color_data_root)
        
        # Color for stems of leaves
        colors_stem = polydata_stem.GetPointData().GetArray("Colors")
        color_data_stem = vtk.vtkUnsignedCharArray()
        color_data_stem.SetName("Colors")
        color_data_stem.SetNumberOfComponents(3)
        
        if colors_stem:
            for j in range(colors_stem.GetNumberOfTuples()):
                r, g, b = colors_stem.GetTuple(j)
                color_data_stem.InsertNextTuple3(r,g,b)
        
        polydata_stem.GetPointData().SetScalars(color_data_stem)
        
        # Mappers
        mapper_root = vtk.vtkPolyDataMapper()
        mapper_root.SetInputData(polydata_root)
        
        mapper_stem = vtk.vtkPolyDataMapper()
        mapper_stem.SetInputData(polydata_stem)
        
        # Actors
        actor_root = vtk.vtkActor()
        actor_root.SetMapper(mapper_root)
        actor_root.GetProperty().SetOpacity(outer_opacity)
        
        actor_stem = vtk.vtkActor()
        actor_stem.SetMapper(mapper_stem)
        actor_stem.GetProperty().SetOpacity(inner_opacity)
        
        # Assembly
        assembly = vtk.vtkAssembly()
        assembly.AddPart(actor_root)
        assembly.AddPart(actor_stem)
        
        renderer.AddActor(assembly)





def main(config):
    """
    Renders a 3D visualization of plant roots based on a series of CSV files
    containing curve and radius information, and generates an iso-surface
    using Marching Cubes.
    """

    # Initialize the existing VTK renderer
    renderer, renderWindow = setup_vtk_renderer()

    # Rendering the surface and volume
    surface_renderer = SurfaceRenderer(config.path_to_test_data + os.sep + config.tiff_file, config.x_offset, config.y_offset, config.z_offset, 1, 3, 0.95, renderer, tiff_binary_file=config.path_to_test_data + os.sep + config.tiff_binary_file, surface_file=config.path_to_test_data + os.sep + config.surface_file)
    #renderer.AddActor(surface_renderer.surface_actor)

    volume_renderer = VolumeRenderer(
        tiff_file=config.path_to_test_data + os.sep + config.tiff_file,
        x_offset=config.x_offset,
        y_offset=config.y_offset,
        z_offset=config.z_offset,
        pixel_size=1,
        opacity_factor=1  
    )
    renderer.AddVolume(volume_renderer.get_volume())

    render_surfaces(renderer,config.path_to_test_data+os.sep+config.leaf_surface_directory,config.path_to_test_data+os.sep+config.root_surface_directory,config.inner_opacity,config.outer_opacity)

    total_vertices, total_triangles = count_vertices_and_triangles(renderer)
    print(f"Total vertices: {total_vertices}")
    print(f"Total triangles: {total_triangles}")

    renderer.ResetCamera()

    camera = renderer.GetActiveCamera()
    camera.Zoom(1)
    center = [0, 0, config.z_offset]
    camera.SetPosition(center[0], center[1] - 5000, center[2] - 10000)
    camera.SetFocalPoint(center[0], center[1], center[2])
    camera.SetViewUp(0, 0, 1)
    camera.SetClippingRange(1000, 60000)


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
