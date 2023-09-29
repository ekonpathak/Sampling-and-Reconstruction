import vtk
import numpy as np
import random
# from vtk.util.numpy_support import vtk_to_numpy
import time
from vtk.util import numpy_support as vtk_np
from scipy.interpolate import griddata
from plotly.subplots import make_subplots
import vtk
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pyvista as pv
import ipywidgets as widgets
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from IPython.display import display
import time
# from vtk.util.numpy_support import numpy_to_vtk
import sys
import numpy as np

para1='nearest'

def simple_random_sampling1(vti_file, sampling_percentage):
    sampling_percentage = sampling_percentage/100
    # Read the vti file
    reader1 = vtk.vtkXMLImageDataReader()
    reader1.SetFileName(vti_file)
    reader1.Update()

    # Get the volume dimensions
    dims = reader1.GetOutput().GetDimensions()
    nx, ny, nz = dims

    # Get the scalar values
    scalar_range = reader1.GetOutput().GetScalarRange()

    # Create an array to store the sampled points
    num_samples = int(nx * ny * nz * sampling_percentage)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(num_samples + 8) # add 8 corner points
    data= vtk.vtkFloatArray()
    data.SetName("Data")
    data.SetNumberOfValues(num_samples + 8)

    # Add the corner points
    corner_points = np.array([[0, 0, 0], [nx-1, 0, 0], [0, ny-1, 0], [nx-1, ny-1, 0], 
                              [0, 0, nz-1], [nx-1, 0, nz-1], [0, ny-1, nz-1], [nx-1, ny-1, nz-1]])
    corner_scalars = np.array([reader1.GetOutput().GetScalarComponentAsFloat(p[0], p[1], p[2], 0) 
                               for p in corner_points])
    for i, p in enumerate(corner_points):
        points.SetPoint(i, p[0], p[1], p[2])
        data.SetValue(i, corner_scalars[i])

    # Sample the volume data
    sampled = np.zeros((nx, ny, nz), dtype=bool)
    num_sampled = 0
    while num_sampled < num_samples:
        x = np.random.randint(0, nx)
        y = np.random.randint(0, ny)
        z = np.random.randint(0, nz)
        if not sampled[x, y, z]:
            sampled[x, y, z] = True
            points.SetPoint(num_sampled+8, x, y, z)
            value = reader1.GetOutput().GetScalarComponentAsFloat(x, y, z, 0)
            data.SetValue(num_sampled+8, value)
            num_sampled += 1

    # Create a polydata object and set the points and scalars
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(data)

    # Write the polydata to a VTKPolyData file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("sampled_output.vtp")
    writer.SetInputData(polydata)
    writer.Write()
    
    return "sampled_output.vtp"

samplingPercentage = int(input("Enter the Percentage: "))
simple_random_sampling1('Isabel_3D.vti', samplingPercentage)

def compute_SNR(arrgt, arr_recon):
    diff = arrgt - arr_recon
    sqd_max_diff = (np.max(arrgt) - np.min(arrgt))**2
    snr = 10 * np.log10(sqd_max_diff / np.mean(diff**2))
    return snr

def reconstruct_volume(method):
    # Load the original volume data
    reader1 = vtk.vtkXMLImageDataReader()
    reader1.SetFileName("Isabel_3D.vti")
    reader1.Update()
    volume_data = reader1.GetOutput()

    # Load the sampled points
    sampled_points_reader1 = vtk.vtkXMLPolyDataReader()
    sampled_points_reader1.SetFileName("sampled_output.vtp")


    sampled_points_reader1.Update()
    sampled_points_data = sampled_points_reader1.GetOutput()

    # Get the points and values from the sampled points data
    sampled_points = vtk.vtkPoints()
    sampled_values = vtk.vtkFloatArray()
    sampled_values.SetNumberOfComponents(1)
    sampled_values.SetName("Data")  # Assume scalar array name is "Data"
    for i in range(sampled_points_data.GetNumberOfPoints()):
        point = sampled_points_data.GetPoint(i)
        value = sampled_points_data.GetPointData().GetArray("Data").GetTuple1(i)  # Use GetTuple1() to get scalar value
        sampled_points.InsertNextPoint(point)
        sampled_values.InsertNextTuple1(value)

    # Convert the points and values to NumPy arrays
    sampled_points_np = vtk_np.vtk_to_numpy(sampled_points.GetData())
    sampled_values_np = vtk_np.vtk_to_numpy(sampled_values)

    # Specify the grid (volume) points using the original volume data
    grid_points = vtk.vtkPoints()
    grid_points.SetDataTypeToFloat()
    grid_points.Allocate(volume_data.GetNumberOfPoints())

    for i in range(volume_data.GetNumberOfPoints()):
        grid_points.InsertNextPoint(volume_data.GetPoint(i))

    # Convert the grid points to a NumPy array
    grid_points_np = vtk_np.vtk_to_numpy(grid_points.GetData())

    if(method =="nearest"):
        #reconstructed_volume_nearest = griddata(sampled_points_np, sampled_values_np, grid_points_np, method='nearest')
        start_time = time.time()  
        RVN = griddata(sampled_points_np, sampled_values_np, grid_points_np, method=method)
        end_time = time.time()
        reconstruction_time = end_time - start_time
        print("time for reconstruction using nearest neighbour where sample percent is ",para1)
        print("time : ",reconstruction_time) 
            # Create a new float array for the reconstructed volume data
        original_volume_np = vtk_np.vtk_to_numpy(volume_data.GetPointData().GetScalars())


        reconstructed_volume_np = RVN

        # Compute SNR
        snr = compute_SNR(original_volume_np, reconstructed_volume_np)
        print("SNR:", snr)

        # Create a new float array for the reconstructed volume data
        RVA = vtk.vtkFloatArray()
        RVA.SetNumberOfComponents(1)
        RVA.SetName("Data")

        # Set the NumPy array to the reconstructed volume array
        RVA.SetArray(RVN.ravel(), RVN.size, 1)

        # Set the reconstructed volume array to the original volume data
        volume_data.GetPointData().SetScalars(RVA)

        # Write the reconstructed volume data to a VTI file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("reconstructed_volume_nearest2.vti")
        writer.SetInputData(volume_data)
        writer.Write()
    else:
        start_time = time.time()  
        RVL = griddata(sampled_points_np, sampled_values_np, grid_points_np, method=method)
        end_time = time.time()
        reconstruction_time = end_time - start_time
        print("time for reconstruction using linear where sample percent is ",para1)
        print("time : ",reconstruction_time) 
        original_np = vtk_np.vtk_to_numpy(volume_data.GetPointData().GetScalars())


        volume_np = RVL

        # Compute SNR
        snr = compute_SNR(original_np, volume_np)
        print("SNR:", snr)
        RVA = vtk.vtkFloatArray()
        RVA.SetNumberOfComponents(1)
        RVA.SetName("Data")
        RVA.SetArray(RVL.ravel().astype(np.float32), RVL.size, 1)

        # Set the reconstructed volume array to the original volume data
        volume_data.GetPointData().SetScalars(RVA)

        # Write the reconstructed volume data to a VTI file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("reconstructed_volume_linear.vti")
        writer.SetInputData(volume_data)
        writer.Write()

            
        # Measure reconstruction time
        
method=input("enter method for reconstruction")           
reconstruct_volume(method)


