import csv
import os
import numpy as np
from scipy.interpolate import splev, splprep, splrep

def get_curve(csv_file, spacing=10):
    """
    Reads a CSV file and returns a resampled list of 3D points that form a curve.

    Args:
        csv_file (str): Path to the CSV file containing the curve data.
        spacing (int, optional): Spacing between points after resampling. Default is 10.

    Returns:
        list: A list of resampled 3D points forming the curve.
    """
    curve_points = []
    with open(csv_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader)  # Skip the header row
        for row in reader:
            curve_points.append([float(row[0]), float(row[1]), float(row[2])])
    
    return resample_curve(curve_points, spacing)


def resample_curve(curve, spacing):
    """
    Resample a curve of 3D points at evenly spaced intervals.

    Args:
        curve (list): A list of 3D points representing the curve.
        spacing (float): The desired spacing between resampled points.

    Returns:
        list: A list of resampled 3D points.
    """
    curve = np.array(curve)
    curve_length = np.sum(np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1)))
    num_points = int(np.ceil(curve_length / spacing))

    tck, u = splprep(curve.T, s=0, per=False)
    u_new = np.linspace(0, 1, num_points)
    return np.column_stack(splev(u_new, tck, der=0)).tolist()

def get_information(information_path):
    """
    Reads a CSV file containing curve and radius information and returns it as a list of rows.

    Args:
        information_path (str): The path to the CSV file containing the information.

    Returns:
        list: A list of rows, where each row is a list of strings representing the curve and radius information.
    """
    information = []
    with open(information_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='|')
        next(reader)  # Skip the header row
        for row in reader:
            information.append(row)
    return information


def get_radius(radius_path, target_lengths):
    """
    Reads a CSV file containing radius values and resamples them to match target lengths.

    Args:
        radius_path (str): Path to the CSV file containing radius data.
        target_lengths (list): List of target lengths to resample the radius data.

    Returns:
        list: A list of resampled radius data chunks.
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

    Args:
        value (str): The string to check.

    Returns:
        bool: True if the string can be converted to a float, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def resample_radius(radius_split, target_lengths):
    """
    Resamples a list of radius data chunks to match target lengths.

    Args:
        radius_split (list): A list of radius data chunks.
        target_lengths (list): A list of target lengths to resample the radius data.

    Returns:
        list: A list of resampled radius data chunks.
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


def read_csv_files(directories, base_path, track_index=False):
    """
    Reads CSV files from a list of directories and sorts them.

    Args:
        directories (list): List of directories containing CSV files.
        base_path (str): Base path to the directories.
        track_index (bool): If True, adds the directory index to each file.

    Returns:
        list: List of found CSV files (with or without directory index).
    """
    all_csv_files = []
    for i, directory in enumerate(directories):
        files = sorted(
            [filename for filename in os.listdir(os.path.join(base_path, directory)) if filename.endswith(".csv")],
            key=lambda x: x.lower()
        )
        if track_index:
            all_csv_files += [[filename, i] for filename in files]  # Adds directory index
        else:
            all_csv_files += files  # Simple list without index
    return all_csv_files


def read_information_file(path):
    """Reads the information file and converts the values to floats."""
    information = get_information(path)
    return [[float(value) for value in sublist] for sublist in information]

def process_curves(entries, directories, base_path, sample, track_index=False):
    """
    Extracts the curves from CSV files and stores their Z coordinates.

    Args:
        entries (list): List of CSV filenames.
        directories (list): List of directories to search for the files.
        base_path (str): Base path to the directories.
        sample (int): Sampling rate for the curves.
        track_index (bool): If True, `entries` contains a list [filename, index].

    Returns:
        tuple: (List of extracted Z coordinates, Dictionary of curves {filename: curve}).
    """
    all_z = []
    curve_points_dict = {}
    for entry in entries:
        if track_index:
            if isinstance(entry, list) and len(entry) == 2:
                filename, i = entry
            else:
                raise ValueError(f"Unexpected format for entry: {entry}")
            directory = directories[i]
        else:
            filename = entry
            directory = next((d for d in directories if os.path.exists(os.path.join(base_path, d, filename))), None)
            if directory is None:
                continue  # Skip if file doesn't exist in the given directories

        csv_path = os.path.join(base_path, directory, filename)
        curve_points = get_curve(csv_path, sample)
        
        if track_index:
            curve_points_dict[filename] = [curve_points, i]
        else:
            curve_points_dict[filename] = curve_points
        
        all_z.append(curve_points[0][2])  # Extract the first Z coordinate

    return all_z, curve_points_dict

def get_sorted_vtk_files(directory):
    """ Récupère et trie les fichiers .vtk d'un dossier."""
    files = [f for f in os.listdir(directory) if f.endswith(".vtk")]
    files.sort()  # Trie les fichiers par ordre alphabétique
    return [os.path.join(directory, f) for f in files]