import vtk
import numpy as np

def load_and_display_vtk(vtk_file):
    # Create a reader for the .vtk file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)

    # Update the reader to load the data
    reader.Update()

    # Get the output from the reader (this is the PolyData)
    polydata = reader.GetOutput()

    # Check if the polydata has colors (point data)
    colors = polydata.GetPointData().GetArray("Colors")  # Assuming the color array is named 'Colors'

    # If colors exist, scale them to the 0-255 range
    if colors:
        print("Colors found in the dataset!")
        color_data = vtk.vtkUnsignedCharArray()
        color_data.SetName("Colors")
        color_data.SetNumberOfComponents(3)  # Indicating that each tuple will have 3 components: RGB

        # Scale the color data to 0-255
        for i in range(colors.GetNumberOfTuples()):
            r, g, b = colors.GetTuple(i)
            color_data.InsertNextTuple3(int(r * 255), int(g * 255), int(b * 255))

        # Associate the color data with the point data
        polydata.GetPointData().SetScalars(color_data)
    
    # Create a mapper to map the PolyData to graphics primitives
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # Create an actor to hold the geometry and make it visible
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer and add the actor to it
    renderer = vtk.vtkRenderer()

    # Create a render window and set the size
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 800)

    # Create a render window interactor to allow user interaction
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the renderer to the render window
    render_window.AddRenderer(renderer)

    # Add the actor to the renderer
    renderer.AddActor(actor)

    # Set a background color (optional)
    renderer.SetBackground(0.0, 0.3, 0.5)
    renderer.ResetCamera()

    # Start the render window interactor
    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    # Specify the path to your .vtk file
    vtk_file_path = "/home/caro/Documents/Ressources/iso_surface_colored.vtk"
    
    # Call the function to load and display the .vtk file
    load_and_display_vtk(vtk_file_path)
