from torch import nn
import os
import torch
from torch.utils.data import Dataset
import nrrd
import numpy as np
from torchvision import transforms

import torch
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import torchvision.transforms as T
from timm import create_model
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import random
import csv
import scipy.ndimage

from torchvision.models.video import r3d_18, mc3_18
import torchvision.models.video as models
import torch.nn.functional as F


def process_fcsv_to_tensor(fcsv_file, output_fcsv=None, return_labels=False):
    """
    Parses an FCSV file to extract physical coordinates (x, y, z) and optional labels.
    Optionally writes extracted data into a new FCSV file.

    Args:
        fcsv_file (str): Path to the FCSV file to read.
        output_fcsv (str, optional): Path to save the processed FCSV file (with headers).
        return_labels (bool): If True, returns both coordinates and labels.

    Returns:
        torch.Tensor: Tensor of shape (num_landmarks, 3) containing physical coordinates.
        list (optional): List of labels corresponding to each landmark.
    """
    # Header for FCSV output
    header = [
        "# Markups fiducial file version = 5.6",
        "# CoordinateSystem = LPS",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
    ]

    # Open and read the FCSV file
    with open(fcsv_file, 'r') as f:
        lines = f.readlines()

    coordinates = []
    labels = [] if return_labels else None
    entries = []  # To store full entries if saving to a new file

    for line in lines:
        # Skip comment lines starting with '#'
        if line.startswith('#'):
            continue

        # Split the line into components
        parts = line.strip().split(',')

        # Ensure the line has enough components for x, y, z
        if len(parts) > 3:
            # Extract x, y, z coordinates
            x, y, z = map(float, parts[1:4])
            coordinates.append((x, y, z))

            # Extract label if required
            label = parts[11] if return_labels and len(parts) > 11 else None
            if return_labels:
                labels.append(label)

            # Store full line for writing output later
            entries.append(parts)

    # Convert coordinates to PyTorch tensor
    coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)

    # Write output FCSV file (optional)
    if output_fcsv:
        with open(output_fcsv, 'w') as f:
            # Write header
            f.write('\n'.join(header) + '\n')

            # Write all entries
            for i, parts in enumerate(entries):
                parts[0] = f"Fiducial_{i+1}"  # Ensure IDs are unique
                f.write(','.join(parts) + '\n')

    if return_labels:
        return coordinates_tensor, labels
    return coordinates_tensor


class NRRDDatasetDynamicSlices(Dataset):
    def __init__(self, folder_path, slice_axis=0, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing .nrrd and .fcsv files.
            slice_axis (int): Axis along which to slice the 3D volume (0 = axial, 1 = coronal, 2 = sagittal).
            transform (callable, optional): Optional transform to apply to the slices.
        """
        self.folder_path = folder_path
        self.slice_axis = slice_axis
        self.transform = transform

        # Match NRRD and FCSV files by patient ID
        self.pairs = self.match_files(folder_path)

        # Calculate total number of slices across all volumes
        self.slices_info = self.calculate_slices_info()

    def match_files(self, folder_path):
        """Matches NRRD and FCSV files in the given folder based on patient ID."""
        nrrd_files = {}
        fcsv_files = {}

        for file_name in os.listdir(folder_path):
            if file_name.endswith('masked.nrrd'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                nrrd_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                fcsv_files[patient_id] = os.path.join(folder_path, file_name)

        matched_pairs = []
        for patient_id in nrrd_files:
            if patient_id in fcsv_files:
                matched_pairs.append((patient_id, nrrd_files[patient_id], fcsv_files[patient_id]))

        return matched_pairs

    def calculate_slices_info(self):
        """Calculate the total number of slices and store slice-to-volume mapping."""
        slices_info = []
        for patient_id, nrrd_path, fcsv_path in self.pairs:
            # Load volume and get shape
            volume, _ = nrrd.read(nrrd_path)
            num_slices = volume.shape[self.slice_axis]

            # Record mapping of slices to volumes
            slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx) for slice_idx in range(num_slices)])

        return slices_info

    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        # Retrieve slice info for the given index
        patient_id, nrrd_path, fcsv_path, slice_idx = self.slices_info[idx]

        # Load volume
        volume, header = nrrd.read(nrrd_path)
        volume = torch.tensor(volume, dtype=torch.float32)

        # Extract slice along the specified axis
        if self.slice_axis == 0:  # Axial
            slice_2d = volume[slice_idx, :, :]
        elif self.slice_axis == 1:  # Coronal
            slice_2d = volume[:, slice_idx, :]
        elif self.slice_axis == 2:  # Sagittal
            slice_2d = volume[:, :, slice_idx]

        # Parse FCSV landmarks and convert to voxel coordinates
        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])
        physical_landmarks, labels = process_fcsv_to_tensor(fcsv_path, return_labels=True)
        voxel_landmarks = (physical_landmarks - origin) / spacing

        # Apply transformations
        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels


class NRRDDatasetMultiAxis(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing .nrrd and .fcsv files.
            transform (callable, optional): Optional transform to apply to the slices.
        """
        self.folder_path = folder_path
        self.transform = transform

        # Match NRRD and FCSV files by patient ID
        self.pairs = self.match_files(folder_path)

        # Store slices for all three axes
        self.slices_info = self.calculate_slices_info()

    def match_files(self, folder_path):
        """Matches NRRD and FCSV files in the given folder based on patient ID."""
        nrrd_files = {}
        fcsv_files = {}

        for file_name in os.listdir(folder_path):
            if file_name.endswith('masked.nrrd'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                nrrd_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                fcsv_files[patient_id] = os.path.join(folder_path, file_name)

        matched_pairs = []
        for patient_id in nrrd_files:
            if patient_id in fcsv_files:
                matched_pairs.append((patient_id, nrrd_files[patient_id], fcsv_files[patient_id]))

        return matched_pairs

    def calculate_slices_info(self):
        """Calculate the total number of slices for all three axes."""
        slices_info = []
        for patient_id, nrrd_path, fcsv_path in self.pairs:
            # Load volume and get shape
            volume, _ = nrrd.read(nrrd_path)
            shape = volume.shape  # (Z, Y, X) => (Depth, Height, Width)

            # Store slice info for all three axes
            for axis in range(3):
                num_slices = shape[axis]
                slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx, axis) for slice_idx in range(num_slices)])

        return slices_info

    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        # Retrieve slice info for the given index
        patient_id, nrrd_path, fcsv_path, slice_idx, slice_axis = self.slices_info[idx]

        # Load volume
        volume, header = nrrd.read(nrrd_path)
        volume = torch.tensor(volume, dtype=torch.float32)

        # Extract slice along the specified axis
        if slice_axis == 0:  # Axial
            slice_2d = volume[slice_idx, :, :]
        elif slice_axis == 1:  # Coronal
            slice_2d = volume[:, slice_idx, :]
        elif slice_axis == 2:  # Sagittal
            slice_2d = volume[:, :, slice_idx]

        # Parse FCSV landmarks and convert to voxel coordinates
        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])
        physical_landmarks, labels = process_fcsv_to_tensor(fcsv_path, return_labels=True)
        voxel_landmarks = (physical_landmarks - origin) / spacing

        # Apply transformations
        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels
    
class NRRDDatasetMultiAxis_test(Dataset):
    def __init__(self, folder_path, transform=None, test_mode=False):
        """
        Args:
            folder_path (str): Path to the folder containing .nrrd and optionally .fcsv files.
            transform (callable, optional): Optional transform to apply to the slices.
            test_mode (bool): If True, allows loading data without .fcsv files.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.test_mode = test_mode
        self.pairs = self.match_files(folder_path)
        self.slices_info = self.calculate_slices_info()

    def match_files(self, folder_path):
        """Matches NRRD and FCSV files in the given folder based on patient ID."""
        nrrd_files = {}
        fcsv_files = {}

        for file_name in os.listdir(folder_path):
            if file_name.endswith('masked.nrrd'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                nrrd_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                fcsv_files[patient_id] = os.path.join(folder_path, file_name)

        matched_pairs = []
        for patient_id, nrrd_path in nrrd_files.items():
            fcsv_path = fcsv_files.get(patient_id, None)  # Get FCSV path if available
            if fcsv_path or self.test_mode:
                matched_pairs.append((patient_id, nrrd_path, fcsv_path))
        
        return matched_pairs

    def calculate_slices_info(self):
        """Calculate the total number of slices for all three axes."""
        slices_info = []
        for patient_id, nrrd_path, fcsv_path in self.pairs:
            volume, _ = nrrd.read(nrrd_path)
            shape = volume.shape  # (Z, Y, X) => (Depth, Height, Width)
            
            for axis in range(3):
                num_slices = shape[axis]
                slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx, axis) for slice_idx in range(num_slices)])

        return slices_info

    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        patient_id, nrrd_path, fcsv_path, slice_idx, slice_axis = self.slices_info[idx]
        volume, header = nrrd.read(nrrd_path)
        volume = torch.tensor(volume, dtype=torch.float32)

        # Extract slice along the specified axis
        if slice_axis == 0:  # Axial
            slice_2d = volume[slice_idx, :, :]
        elif slice_axis == 1:  # Coronal
            slice_2d = volume[:, slice_idx, :]
        elif slice_axis == 2:  # Sagittal
            slice_2d = volume[:, :, slice_idx]

        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])

        if fcsv_path:  # Only process landmarks if FCSV exists
            physical_landmarks, labels = process_fcsv_to_tensor(fcsv_path, return_labels=True)
            voxel_landmarks = (physical_landmarks - origin) / spacing
        else:
            voxel_landmarks = torch.empty((0, 3))  # Empty tensor for missing landmarks
            physical_landmarks = torch.empty((0, 3))
            labels = []

        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels


class NRRDDatasetMultiAxis_diag1(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing .nrrd and .fcsv files.
            transform (callable, optional): Optional transform to apply to the slices.
        """
        self.folder_path = folder_path
        self.transform = transform

        # Match NRRD and FCSV files by patient ID
        self.pairs = self.match_files(folder_path)

        # Store slices for all three axes and one diagonal slice
        self.slices_info = self.calculate_slices_info()

    def match_files(self, folder_path):
        """Matches NRRD and FCSV files in the given folder based on patient ID."""
        nrrd_files = {}
        fcsv_files = {}

        for file_name in os.listdir(folder_path):
            if file_name.endswith('masked.nrrd'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                nrrd_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                fcsv_files[patient_id] = os.path.join(folder_path, file_name)

        matched_pairs = []
        for patient_id in nrrd_files:
            if patient_id in fcsv_files:
                matched_pairs.append((patient_id, nrrd_files[patient_id], fcsv_files[patient_id]))

        return matched_pairs

    def calculate_slices_info(self):
        """Calculate the total number of slices for all three axes and one diagonal slice."""
        slices_info = []
        for patient_id, nrrd_path, fcsv_path in self.pairs:
            # Load volume and get shape
            volume, _ = nrrd.read(nrrd_path)
            shape = volume.shape  # (Z, Y, X) => (Depth, Height, Width)

            # Standard Axial (Z), Coronal (Y), and Sagittal (X) slices
            for axis in range(3):
                num_slices = shape[axis]
                slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx, axis) for slice_idx in range(num_slices)])

            # Include only one diagonal slice
            num_diagonal_slices = min(shape)  # Diagonal slices limited by smallest dimension
            for slice_idx in range(num_diagonal_slices):
                slices_info.append((patient_id, nrrd_path, fcsv_path, slice_idx, "diagonal1"))

        return slices_info

    def extract_diagonal_slice(self, volume, slice_idx):
            """
            Extracts a full 2D diagonal slice from the 3D volume.

            Args:
                volume (numpy array): 3D volume of shape (Z, Y, X).
                slice_idx (int): Index for slicing along the diagonal axis.

            Returns:
                diag_slice (numpy array): Extracted 2D diagonal slice.
            """
            shape = volume.shape  # (Z, Y, X)
            
            # Determine the maximum possible diagonal slice size
            diag_size = min(shape[0], shape[1], shape[2])

            # Initialize a 2D slice
            diag_slice = np.zeros((diag_size, diag_size))

            for i in range(diag_size):
                for j in range(diag_size):
                    x_idx = i + j + slice_idx  # Create a full diagonal step in X direction
                    if x_idx < shape[2]:  # Ensure bounds in X
                        diag_slice[i, j] = volume[i, j, x_idx]

            return diag_slice  # Shape (diag_size, diag_size)



    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        # Retrieve slice info for the given index
        patient_id, nrrd_path, fcsv_path, slice_idx, slice_axis = self.slices_info[idx]

        # Load volume
        volume, header = nrrd.read(nrrd_path)
        volume = torch.tensor(volume, dtype=torch.float32)

        # Extract slice based on axis type
        if slice_axis == 0:  # Axial
            slice_2d = volume[slice_idx, :, :]
        elif slice_axis == 1:  # Coronal
            slice_2d = volume[:, slice_idx, :]
        elif slice_axis == 2:  # Sagittal
            slice_2d = volume[:, :, slice_idx]
        elif slice_axis == "diagonal1":  # Only one diagonal slice
            slice_2d = self.extract_diagonal_slice(volume.numpy(), slice_idx)
            slice_2d = torch.tensor(slice_2d, dtype=torch.float32)
        else:
            raise ValueError("Invalid slice axis")

        # Parse FCSV landmarks and convert to voxel coordinates
        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])
        physical_landmarks, labels = process_fcsv_to_tensor(fcsv_path, return_labels=True)
        voxel_landmarks = (physical_landmarks - origin) / spacing

        # Apply transformations
        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels
    



class NRRDDatasetMultiAxis_diag1_rotated(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.pairs = self.match_files(folder_path)
        self.slices_info = self.calculate_slices_info()

    def match_files(self, folder_path):
        """Matches NRRD and FCSV files in the given folder based on patient ID."""
        nrrd_files = {}
        fcsv_files = {}

        for file_name in os.listdir(folder_path):
            if file_name.endswith('masked.nrrd'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                nrrd_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]
                fcsv_files[patient_id] = os.path.join(folder_path, file_name)

        matched_pairs = []
        for patient_id in nrrd_files:
            if patient_id in fcsv_files:
                matched_pairs.append((patient_id, nrrd_files[patient_id], fcsv_files[patient_id]))

        return matched_pairs

    def calculate_slices_info(self):
        """Calculate slice indices for standard axes and rotated volume (for diagonal slices)."""
        slices_info = []
        for patient_id, nrrd_path, fcsv_path in self.pairs:
            volume, _ = nrrd.read(nrrd_path)
            shape = volume.shape  # (Z, Y, X)

            # Standard Axial (Z), Coronal (Y), Sagittal (X) slices
            for axis in range(3):
                num_slices = shape[axis]
                slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx, axis) for slice_idx in range(num_slices)])

            # Rotated Axial slices
            slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx, "rotated_axial") for slice_idx in range(shape[0])])

        return slices_info

    def rotate_volume_45(self, volume):
        """Rotates the 3D volume by 45 degrees around the Y-axis (sagittal rotation)."""
        rotated_volume = scipy.ndimage.rotate(volume, angle=45, axes=(2, 0), reshape=False, mode='nearest')
        return rotated_volume

    def rotate_landmarks_45(self, landmarks, origin):
        """Applies a 45-degree rotation around the Y-axis to the physical landmark coordinates."""
        theta = np.radians(45)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Ensure landmarks are NumPy array
        landmarks_np = landmarks.numpy() if isinstance(landmarks, torch.Tensor) else landmarks

        # Translate landmarks to origin, apply rotation, and translate back
        rotated_landmarks = (rotation_matrix @ (landmarks_np - origin).T).T + origin

        return torch.tensor(rotated_landmarks, dtype=torch.float32)  # Convert back to Tensor

    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        patient_id, nrrd_path, fcsv_path, slice_idx, slice_axis = self.slices_info[idx]

        volume, header = nrrd.read(nrrd_path)
        volume = torch.tensor(volume, dtype=torch.float32)

        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])

        # Process landmarks
        physical_landmarks, labels = process_fcsv_to_tensor(fcsv_path, return_labels=True)
        voxel_landmarks = (physical_landmarks - origin) / spacing

        # Extract slice based on axis
        if slice_axis == 0:  # Axial
            slice_2d = volume[slice_idx, :, :]
        elif slice_axis == 1:  # Coronal
            slice_2d = volume[:, slice_idx, :]
        elif slice_axis == 2:  # Sagittal
            slice_2d = volume[:, :, slice_idx]
        elif slice_axis == "rotated_axial":
            rotated_volume = self.rotate_volume_45(volume.numpy())  # Rotate volume
            slice_2d = torch.tensor(rotated_volume[slice_idx, :, :], dtype=torch.float32)

            # Rotate landmarks
            rotated_physical_landmarks = self.rotate_landmarks_45(physical_landmarks, origin)
            voxel_landmarks = (rotated_physical_landmarks - origin) / spacing
        else:
            raise ValueError("Invalid slice axis")

        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels, str(slice_axis)



class NRRDDatasetMultiAxis_diag1_2(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing .nrrd and .fcsv files.
            transform (callable, optional): Optional transform to apply to the slices.
        """
        self.folder_path = folder_path
        self.transform = transform

        # Match NRRD and FCSV files by patient ID
        self.pairs = self.match_files(folder_path)

        # Store slices for all three axes and diagonal slices
        self.slices_info = self.calculate_slices_info()

    def match_files(self, folder_path):
        """Matches NRRD and FCSV files in the given folder based on patient ID."""
        nrrd_files = {}
        fcsv_files = {}

        for file_name in os.listdir(folder_path):
            if file_name.endswith('masked.nrrd'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                nrrd_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                fcsv_files[patient_id] = os.path.join(folder_path, file_name)

        matched_pairs = []
        for patient_id in nrrd_files:
            if patient_id in fcsv_files:
                matched_pairs.append((patient_id, nrrd_files[patient_id], fcsv_files[patient_id]))

        return matched_pairs

    def calculate_slices_info(self):
        """Calculate the total number of slices for all three axes and diagonal slices."""
        slices_info = []
        for patient_id, nrrd_path, fcsv_path in self.pairs:
            # Load volume and get shape
            volume, _ = nrrd.read(nrrd_path)
            shape = volume.shape  # (Z, Y, X) => (Depth, Height, Width)

            # Standard Axial (Z), Coronal (Y), and Sagittal (X) slices
            for axis in range(3):
                num_slices = shape[axis]
                slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx, axis) for slice_idx in range(num_slices)])

            # Always include diagonal slices
            num_diagonal_slices = min(shape)  # Diagonal slices limited by smallest dimension
            for slice_idx in range(num_diagonal_slices):
                slices_info.append((patient_id, nrrd_path, fcsv_path, slice_idx, "diagonal1"))
                slices_info.append((patient_id, nrrd_path, fcsv_path, slice_idx, "diagonal2"))

        return slices_info

    def extract_diagonal_slice(self, volume, slice_idx, diagonal_type):
        """
        Extracts a 2D diagonal slice from the 3D volume.

        Args:
            volume (numpy array): 3D volume of shape (Z, Y, X).
            slice_idx (int): Index for slicing along the diagonal.
            diagonal_type (str): "diagonal1" (main diagonal), "diagonal2" (reverse diagonal).

        Returns:
            diag_slice (numpy array): Extracted 2D diagonal slice.
        """
        shape = volume.shape  # (Z, Y, X)

        # Determine slice size (limit by smallest dimension)
        diag_size = min(shape[0], shape[1], shape[2])

        # Initialize the 2D slice
        diag_slice = np.zeros((diag_size, diag_size))

        for i in range(diag_size):
            for j in range(diag_size):
                if diagonal_type == "diagonal1":
                    x_idx = j + slice_idx  # Move right in X direction
                    if x_idx < shape[2]:  # Ensure within bounds
                        diag_slice[i, j] = volume[i, j, x_idx]

                elif diagonal_type == "diagonal2":
                    x_idx = shape[2] - j - 1 - slice_idx  # Move left in X direction
                    if 0 <= x_idx < shape[2]:  # Ensure within bounds
                        diag_slice[i, j] = volume[i, j, x_idx]

                else:
                    raise ValueError("Invalid diagonal type. Use 'diagonal1' or 'diagonal2'.")

        return diag_slice  # Returns a proper 2D slice



    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        # Retrieve slice info for the given index
        patient_id, nrrd_path, fcsv_path, slice_idx, slice_axis = self.slices_info[idx]

        # Load volume
        volume, header = nrrd.read(nrrd_path)
        volume = torch.tensor(volume, dtype=torch.float32)

        # Extract slice based on axis type
        if slice_axis == 0:  # Axial
            slice_2d = volume[slice_idx, :, :]
        elif slice_axis == 1:  # Coronal
            slice_2d = volume[:, slice_idx, :]
        elif slice_axis == 2:  # Sagittal
            slice_2d = volume[:, :, slice_idx]
        elif slice_axis in ["diagonal1", "diagonal2"]:  # Diagonal
            slice_2d = self.extract_diagonal_slice(volume.numpy(), slice_idx, slice_axis)
            slice_2d = torch.tensor(slice_2d, dtype=torch.float32)
        else:
            raise ValueError("Invalid slice axis")

        # Parse FCSV landmarks and convert to voxel coordinates
        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])
        physical_landmarks, labels = process_fcsv_to_tensor(fcsv_path, return_labels=True)
        voxel_landmarks = (physical_landmarks - origin) / spacing

        # Apply transformations
        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels


# def process_fcsv_to_tensor(fcsv_file):
#     """
#     Parses an FCSV file to extract physical coordinates (x, y, z) as a PyTorch tensor.
    
#     Args:
#         fcsv_file (str): Path to the FCSV file.
        
#     Returns:
#         torch.Tensor: Tensor of shape (num_landmarks, 3) containing physical coordinates.
#     """
#     # Open and read the FCSV file
#     with open(fcsv_file, 'r') as f:
#         lines = f.readlines()

#     # List to store coordinates
#     coordinates = []

#     for line in lines:
#         # Skip comment lines starting with '#'
#         if line.startswith('#'):
#             continue

#         # Split the line into components
#         parts = line.strip().split(',')

#         # Ensure the line has enough components for x, y, z
#         if len(parts) > 3:
#             # Extract x, y, z coordinates
#             x, y, z = map(float, parts[1:4])
#             coordinates.append((x, y, z))

#     # Convert to PyTorch tensor
#     return torch.tensor(coordinates, dtype=torch.float32)
# class NRRDDatasetDynamicSlices(Dataset):
#     def __init__(self, folder_path, slice_axis=0, transform=None):
#         """
#         Args:
#             folder_path (str): Path to the folder containing .nrrd and .fcsv files.
#             slice_axis (int): Axis along which to slice the 3D volume (0 = axial, 1 = coronal, 2 = sagittal).
#             transform (callable, optional): Optional transform to apply to the slices.
#         """
#         self.folder_path = folder_path
#         self.slice_axis = slice_axis
#         self.transform = transform

#         # Match NRRD and FCSV files by patient ID
#         self.pairs = self.match_files(folder_path)

#         # Calculate total number of slices across all volumes
#         self.slices_info = self.calculate_slices_info()

#     def match_files(self, folder_path):
#         """Matches NRRD and FCSV files in the given folder based on patient ID."""
#         nrrd_files = {}
#         fcsv_files = {}

#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('.nrrd'):
#                 patient_id = file_name.split('_')[0]  # Extract patient ID
#                 nrrd_files[patient_id] = os.path.join(folder_path, file_name)
#             elif file_name.endswith('.fcsv'):
#                 patient_id = file_name.split('_')[0]  # Extract patient ID
#                 fcsv_files[patient_id] = os.path.join(folder_path, file_name)

#         matched_pairs = []
#         for patient_id in nrrd_files:
#             if patient_id in fcsv_files:
#                 matched_pairs.append((patient_id, nrrd_files[patient_id], fcsv_files[patient_id]))

#         return matched_pairs

#     def calculate_slices_info(self):
#         """Calculate the total number of slices and store slice-to-volume mapping."""
#         slices_info = []
#         for patient_id, nrrd_path, fcsv_path in self.pairs:
#             # Load volume and get shape
#             volume, _ = nrrd.read(nrrd_path)
#             num_slices = volume.shape[self.slice_axis]

#             # Record mapping of slices to volumes
#             slices_info.extend([(patient_id, nrrd_path, fcsv_path, slice_idx) for slice_idx in range(num_slices)])

#         return slices_info

#     def __len__(self):
#         return len(self.slices_info)

#     def __getitem__(self, idx):
#         # Retrieve slice info for the given index
#         patient_id, nrrd_path, fcsv_path, slice_idx = self.slices_info[idx]

#         # Load volume
#         volume, header = nrrd.read(nrrd_path)
#         volume = torch.tensor(volume, dtype=torch.float32)

#         # Extract slice along the specified axis
#         if self.slice_axis == 0:  # Axial
#             slice_2d = volume[slice_idx, :, :]
#         elif self.slice_axis == 1:  # Coronal
#             slice_2d = volume[:, slice_idx, :]
#         elif self.slice_axis == 2:  # Sagittal
#             slice_2d = volume[:, :, slice_idx]

#         # Parse FCSV landmarks and convert to voxel coordinates
#         spacing = np.linalg.norm(header['space directions'], axis=0)
#         origin = np.array(header['space origin'])
#         physical_landmarks = process_fcsv_to_tensor(fcsv_path)
#         voxel_landmarks = (physical_landmarks - origin) / spacing

#         # Apply transformations
#         if self.transform:
#             slice_2d = self.transform(slice_2d)

        


#         return slice_2d, voxel_landmarks, patient_id, slice_idx,spacing, origin,physical_landmarks


def split_dataset_by_volume(dataset, train_volumes, val_volumes):
    """
    Splits a dataset into train and validation sets based on volumes.
    
    Args:
        dataset (Dataset): The full dataset.
        train_volumes (int): Number of volumes for training.
        val_volumes (int): Number of volumes for validation.
    
    Returns:
        tuple: Train and validation datasets.
    """
    # Extract unique volumes from the dataset
    unique_volumes = list(set(info[0] for info in dataset.slices_info))  # patient_id is at index 0
    random.shuffle(unique_volumes)  # Shuffle to randomize the split

    # Ensure the specified number of volumes matches the dataset
    assert train_volumes + val_volumes <= len(unique_volumes), \
        "The sum of train and val volumes must not exceed the total number of unique volumes."

    # Split the volumes
    train_volume_ids = unique_volumes[:train_volumes]
    val_volume_ids = unique_volumes[train_volumes:train_volumes + val_volumes]

    # Filter slices by volume IDs
    train_indices = [idx for idx, info in enumerate(dataset.slices_info) if info[0] in train_volume_ids]
    val_indices = [idx for idx, info in enumerate(dataset.slices_info) if info[0] in val_volume_ids]

    # Create subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


def train_and_evaluate_model(
    model, train_loader, val_loader, device, epochs=30,
    results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results.txt"
):
    """
    Train and evaluate a Swin Transformer model for regression tasks.

    Args:
        model: Swin Transformer model with regression head.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to train on (e.g., "cuda" or "cpu").
        epochs: Number of training epochs.
        results_file: File path to save training and validation results.

    Returns:
        best_loss: Best validation loss achieved during training.
        mse: Mean Squared Error for the best validation epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    best_loss = float("inf")  # Track the best validation loss

    # Open the results file for logging
    with open(results_file, "a") as f:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for images, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks,slice_axis in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(device), labels.to(device).float() 
                
                # Flatten labels to match outputs
                labels = labels.view(labels.size(0), -1)  # Shape: (batch_size, num_landmarks * 3)
                
                optimizer.zero_grad()
                outputs = model(images)  # Shape: (batch_size, num_landmarks * 3)
                # Convert voxel coordinates to physical space
                # Reshape outputs to [batch_size, num_landmarks, 3]
                num_landmarks = 98  # Update this to the correct number of landmarks
                outputs = outputs.view(outputs.size(0), num_landmarks, 3)  # [batch_size, 98, 3]
                spacing = spacing.to(outputs.device)  # Move spacing to the same device as outputs
                origin = origin.to(outputs.device)    # Move origin to the same device as outputs
                
                # print(f"Labels shape (before reshape): {labels.shape}")
                labels = labels.view(labels.size(0), num_landmarks, 3)
                # print(f"Labels shape (after reshape): {labels.shape}")
                # print(f"Spacing shape: {spacing.unsqueeze(1).shape}")
                # print(f"Origin shape: {origin.unsqueeze(1).shape}")


                predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)
                
                # Print voxel and physical coordinates for the first patient along with patient_id
                # if epoch == 0:  # Only print during the first epoch
                #     first_patient_id = patient_id[0]  # Patient ID of the first patient in the batch
                #     patient_voxel_coords = labels[0].view(-1, 3)[:5]  # First 5 landmarks in voxel space
                #     patient_physical_coords = labels_physical[0].view(-1, 3)[:5]  # Corresponding physical coordinates

                    # print("First Patient before prediction (Physical Coordinates):")
                    # print(physical_landmarks.cpu().numpy())
                    # print(f"\nPatient ID: {first_patient_id}")
                    # print("First Patient (Voxel Coordinates):")
                    # print(patient_voxel_coords.cpu().numpy())
                    # print("First Patient after prediction (Physical Coordinates):")
                    # print(patient_physical_coords.cpu().numpy())
                
                loss = criterion(predicted_physical, labels_physical)  # MSE Loss
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}\n")

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks,slice_axis in val_loader:
                    images, labels = images.to(device), labels.to(device).float() 
                    
                    # Flatten labels to match outputs
                    labels = labels.view(labels.size(0), -1)  # Shape: (batch_size, num_landmarks * 3)
                    
                    outputs = model(images)  # Shape: (batch_size, num_landmarks * 3)
                    # Convert voxel coordinates to physical space
                    # Reshape outputs to [batch_size, num_landmarks, 3]
                    num_landmarks = 98  # Update this to the correct number of landmarks
                    outputs = outputs.view(outputs.size(0), num_landmarks, 3)  # [batch_size, 98, 3]
                    spacing = spacing.to(outputs.device)  # Move spacing to the same device as outputs
                    origin = origin.to(outputs.device) 
                  
                    labels = labels.view(labels.size(0), num_landmarks, 3)
                   
                    # Perform the physical coordinates transformation
                    predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                    labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)
                    
                    loss = criterion(predicted_physical, labels_physical)
                    val_loss += loss.item()

                    # Collect predictions and ground truth for metrics
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}\n")
            
            # Save the best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/model/resenet101_diagonal_6.pth")

            
            # Compute and log additional metrics (e.g., MSE)
            all_labels_flat = np.array(all_labels).reshape(-1, 3)  # [batch_size * num_landmarks, 3]
            all_preds_flat = np.array(all_preds).reshape(-1, 3)    # [batch_size * num_landmarks, 3]
            mse = mean_squared_error(all_labels_flat, all_preds_flat)
            print(f"Mean Squared Error (Validation): {mse:.4f}")
            f.write(f"Mean Squared Error (Validation): {mse:.4f}\n")

    return best_loss, mse



def train_and_evaluate_model_save_pred(
    model, train_loader, val_loader, device, epochs=30,
    results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results.txt",
    best_pred_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/best_model_predictions/"
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.MSELoss()
    best_loss = float("inf")

    os.makedirs(best_pred_folder, exist_ok=True)

    with open(results_file, "a") as f:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_patient_landmarks = {}  # Store training predictions
            
            # Training loop
            for images, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(device), labels.to(device).float()
                labels = labels.view(labels.size(0), -1)
                optimizer.zero_grad()
                outputs = model(images)
                num_landmarks = 98
                outputs = outputs.view(outputs.size(0), num_landmarks, 3)
                spacing = spacing.to(outputs.device)
                origin = origin.to(outputs.device)
                labels = labels.view(labels.size(0), num_landmarks, 3)

                predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)
                loss = criterion(predicted_physical, labels_physical)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

                # Store training predictions
                for i in range(images.shape[0]):
                    patient_name = patient_id[i]  # Unique training patient ID
                    train_patient_landmarks[patient_name] = predicted_physical[i].detach().cpu().numpy()

            
            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}\n")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            val_patient_landmarks = {}  # Store validation predictions
            
            with torch.no_grad():
                for images, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in val_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    labels = labels.view(labels.size(0), -1)
                    outputs = model(images)
                    outputs = outputs.view(outputs.size(0), num_landmarks, 3)
                    spacing = spacing.to(outputs.device)
                    origin = origin.to(outputs.device)
                    labels = labels.view(labels.size(0), num_landmarks, 3)
                    predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                    labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)
                    loss = criterion(predicted_physical, labels_physical)
                    val_loss += loss.item()
                    # Collect predictions and ground truth for metrics
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    # Store validation predictions
                    for i in range(images.shape[0]):
                        patient_name = patient_id[i]  # Unique validation patient ID
                        val_patient_landmarks[patient_name] = predicted_physical[i].detach().cpu().numpy()

            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}\n")
            
            # Save model and predictions if validation loss improves
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/model/resnet101_6_pred.pth")
                
                # Save both training and validation predictions
                save_predictions(best_pred_folder, train_patient_landmarks)
                save_predictions(best_pred_folder, val_patient_landmarks)
            all_labels_flat = np.array(all_labels).reshape(-1, 3)  # [batch_size * num_landmarks, 3]
            all_preds_flat = np.array(all_preds).reshape(-1, 3)    # [batch_size * num_landmarks, 3]
            mse = mean_squared_error(all_labels_flat, all_preds_flat)
            print(f"Mean Squared Error (Validation): {mse:.4f}")
            f.write(f"Mean Squared Error (Validation): {mse:.4f}\n")
# Function to save predictions as .fcsv files
def save_predictions(folder, patient_landmarks):
    for patient_id, landmarks in patient_landmarks.items():
        fcsv_path = os.path.join(folder, f"{patient_id}_pred.fcsv")
        labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data/{patient_id}_Fiducial_template_ALL.fcsv"

        header = [
            "# Markups fiducial file version = 5.6",
            "# CoordinateSystem = LPS",
            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
        ]
        
        labels = []
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as file:
                for line in file:
                    if not line.startswith("#") and line.strip():
                        parts = line.strip().split(',')
                        if len(parts) > 11:
                            labels.append(parts[11])

        with open(fcsv_path, mode="w", newline="") as fcsv_file:
            writer = csv.writer(fcsv_file, delimiter=',')
            for line in header:
                fcsv_file.write(line + "\n")
            for i, landmark in enumerate(landmarks):
                label = labels[i] if i < len(labels) else f"{i}"
                writer.writerow([i, *landmark, 0, 0, 0, 0, 1, 1, 0, label, "", ""])


def test_for_voxel(
    model, 
    test_loader, 
    device, 
    results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results_test.txt", 
    output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/output_test_results"
):
    """
    Test the model on the test dataset and save results.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        epochs: Number of testing epochs (for repeated inference, if needed).
        results_file: Path to save overall testing results (loss and metrics).
        output_folder: Folder to save individual patient CSV files with predictions.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    print(f"Testing started...")
    test_loss = 0.0

    with open(results_file, "w") as f:
     
        with torch.no_grad():
            # Dictionary to group results by patient ID
            patient_landmarks = {}

            for images, labels, patient_ids, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(device), labels.to(device).float()

                # Predict voxel coordinates
                outputs = model(images)  # Shape: (batch_size, num_landmarks * 3)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                for i in range(images.size(0)):
                    current_patient_id = patient_id[i]
                    current_spacing = spacing[i].cpu().numpy()
                    current_origin = origin[i].cpu().numpy()
                    current_outputs = outputs[i].cpu().numpy().reshape(-1, 3)  # (num_landmarks, 3)

                    # Convert voxel coordinates to physical coordinates
                    predicted_physical = current_outputs * current_spacing + current_origin

                    # Store final physical coordinates for each patient
                    if current_patient_id not in patient_landmarks:
                        patient_landmarks[current_patient_id] = predicted_physical
                    else:
                        # Aggregate across slices (but here it's final coordinates for the volume)
                        patient_landmarks[current_patient_id] = predicted_physical  # Overwrite, as we want the final result

            # Write each patient's physical landmarks to a CSV
            for patient_id, landmarks in patient_landmarks.items():
                patient_fcsv_path = os.path.join(output_folder, f"{patient_id}.fcsv")
                labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data/{patient_id}_Fiducial_template_ALL.fcsv"
                print("Patient ID:", patient_id)
                
                # Define the header lines
                header = [
                    "# Markups fiducial file version = 5.6",
                    "# CoordinateSystem = LPS",
                    "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
                ]
                
                # Read label information from the provided labels file
                labels = []
                if os.path.exists(labels_file):
                    with open(labels_file, 'r') as file:
                        for line in file:
                            if not line.startswith("#") and line.strip():  # Ignore comments and empty lines
                                parts = line.strip().split(',')
                                if len(parts) > 11:  # Ensure enough columns exist to fetch the label
                                    labels.append(parts[11])  # The label is at index 11
                
                # Write to the .fcsv file
                with open(patient_fcsv_path, mode="w", newline="") as fcsv_file:
                    writer = csv.writer(fcsv_file, delimiter=',')

                    # Write headers
                    for line in header:
                        fcsv_file.write(line + "\n")
                    
                    # Write landmark rows with labels
                    for i, landmark in enumerate(landmarks):
                        label = labels[i] if i < len(labels) else f"Landmark_{i}"  # Assign default if missing
                        writer.writerow([i, *landmark, 0, 0, 0, 0, 1, 1, 0, label, "", ""])

        avg_test_loss = test_loss / len(test_loader)
        print(f"Testing Completed. Test Loss: {avg_test_loss:.4f}")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")

    print(f"Testing complete. Results saved to {results_file} and individual patient files in {output_folder}.")




def test_for_physical_old(
    model, 
    test_loader, 
    device, 
    results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results_test.txt", 
    output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/output_test_results"
):
    """
    Test the model on the test dataset and save results.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        results_file: Path to save overall testing results (loss and metrics).
        output_folder: Folder to save individual patient CSV files with predictions.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    print(f"Testing started...")
    test_loss = 0.0

    with open(results_file, "w") as f:
        with torch.no_grad():
            # Dictionary to group results by patient ID
            patient_landmarks = {}

            for images, labels, patient_ids, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(device), labels.to(device).float()

                # Flatten labels to match outputs
                labels = labels.view(labels.size(0), -1)  # Shape: (batch_size, num_landmarks * 3)

                # Model prediction
                outputs = model(images)  # Shape: (batch_size, num_landmarks * 3)
                
                num_landmarks = 98  # Update this to the correct number of landmarks
                outputs = outputs.view(outputs.size(0), num_landmarks, 3)  # [batch_size, 98, 3]
                spacing = spacing.to(outputs.device)  # Move spacing to the same device as outputs
                origin = origin.to(outputs.device)  
                labels = labels.view(labels.size(0), num_landmarks, 3)

                # Convert voxel coordinates to physical coordinates
                predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)  # (batch_size, num_landmarks, 3)
                labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)

                # Compute loss in physical space
                loss = criterion(predicted_physical, labels_physical)
                test_loss += loss.item()

                for i in range(images.size(0)):
                    current_patient_id = patient_ids[i]

                    # Save physical coordinates for the current patient
                    patient_landmarks[current_patient_id] = predicted_physical[i].cpu().numpy()

            # Write each patient's physical landmarks and labels to a CSV
            for patient_id, landmarks in patient_landmarks.items():
                patient_fcsv_path = os.path.join(output_folder, f"{patient_id}.fcsv")
                labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data_test/NG4120_Fiducial_template_ALL.fcsv"
                #labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data_test/{patient_id}_Fiducial_template_ALL.fcsv"
                print("Patient ID:", patient_id)
                
                # Define the header lines
                header = [
                    "# Markups fiducial file version = 5.6",
                    "# CoordinateSystem = LPS",
                    "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
                ]
                
                # Read label information from the provided labels file
                labels = []
                if os.path.exists(labels_file):
                    with open(labels_file, 'r') as file:
                        for line in file:
                            if not line.startswith("#") and line.strip():  # Ignore comments and empty lines
                                parts = line.strip().split(',')
                                if len(parts) > 11:  # Ensure enough columns exist to fetch the label
                                    labels.append(parts[11])  # The label is at index 11
                
                # Write to the .fcsv file
                with open(patient_fcsv_path, mode="w", newline="") as fcsv_file:
                    writer = csv.writer(fcsv_file, delimiter=',')

                    # Write headers
                    for line in header:
                        fcsv_file.write(line + "\n")
                    
                    # Write landmark rows with labels
                    for i, landmark in enumerate(landmarks):
                        label = labels[i] if i < len(labels) else f"Landmark_{i}"  # Assign default if missing
                        writer.writerow([i, *landmark, 0, 0, 0, 0, 1, 1, 0, label, "", ""])

        avg_test_loss = test_loss / len(test_loader)
        print(f"Testing Completed. Test Loss: {avg_test_loss:.4f}")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")

    print(f"Testing complete. Results saved to {results_file} and individual patient files in {output_folder}.")

def test_for_physical(
    model, 
    test_loader, 
    device, 
    output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/output_test_results"
):
    """
    Test the model on the test dataset and extract volume-related information.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        output_folder: Folder to save individual patient results.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

    model = model.to(device)
    model.eval()
    print(f"Testing started...")

    with torch.no_grad():
        # Dictionary to group results by patient ID
        patient_landmarks = {}

        for images, labelss, patient_ids, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)

            # Model prediction
            outputs = model(images)  # Shape: (batch_size, num_landmarks * 3)
            
            num_landmarks = 98  # Update this to the correct number of landmarks
            outputs = outputs.view(outputs.size(0), num_landmarks, 3)  # [batch_size, 98, 3]
            spacing = spacing.to(outputs.device)  # Move spacing to the same device as outputs
            origin = origin.to(outputs.device)  

            # Convert voxel coordinates to physical coordinates
            predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)  # (batch_size, num_landmarks, 3)

            for i in range(images.size(0)):
                current_patient_id = patient_ids[i]
                
                # Save physical coordinates for the current patient
                patient_landmarks[current_patient_id] = predicted_physical[i].cpu().numpy()

     

        # Write each patient's physical landmarks and labels to a CSV
        for patient_id, landmarks in patient_landmarks.items():
            patient_fcsv_path = os.path.join(output_folder, f"{patient_id}.fcsv")
            labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data_test/NG4120_Fiducial_template_ALL.fcsv"
            #labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data_test/{patient_id}_Fiducial_template_ALL.fcsv"
            print("Patient ID:", patient_id)
            
            # Define the header lines
            header = [
                "# Markups fiducial file version = 5.6",
                "# CoordinateSystem = LPS",
                "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
            ]
            
            # Read label information from the provided labels file
            labels = []
            if os.path.exists(labels_file):
                with open(labels_file, 'r') as file:
                    for line in file:
                        if not line.startswith("#") and line.strip():  # Ignore comments and empty lines
                            parts = line.strip().split(',')
                            if len(parts) > 11:  # Ensure enough columns exist to fetch the label
                                labels.append(parts[11])  # The label is at index 11
            
            # Write to the .fcsv file
            with open(patient_fcsv_path, mode="w", newline="") as fcsv_file:
                writer = csv.writer(fcsv_file, delimiter=',')

                # Write headers
                for line in header:
                    fcsv_file.write(line + "\n")
                
                # Write landmark rows with labels
                for i, landmark in enumerate(landmarks):
                    label = labels[i] if i < len(labels) else f"Landmark_{i}"  # Assign default if missing
                    writer.writerow([i, *landmark, 0, 0, 0, 0, 1, 1, 0, label, "", ""])

    print(f"Testing complete. Results saved to {output_folder}.")

def reverse_rotate_landmarks_45(landmarks, origin):
    """Reverses the 45-degree rotation around the Y-axis to get back to the original space."""
    theta = np.radians(-45)  # Reverse rotation
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # Ensure landmarks are a NumPy array
    landmarks_np = landmarks.cpu().numpy() if isinstance(landmarks, torch.Tensor) else landmarks

    # Apply reverse rotation
    rotated_back_landmarks = (rotation_matrix @ (landmarks_np - origin).T).T + origin

    return torch.tensor(rotated_back_landmarks, dtype=torch.float32)  # Convert back to Tensor

def test_for_physical_rotated(
    model, 
    test_loader, 
    device, 
    results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results_test.txt", 
    output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/output_test_results"
):
    os.makedirs(output_folder, exist_ok=True)
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    print(f"Testing started...")
    test_loss = 0.0

    with open(results_file, "w") as f:
        with torch.no_grad():
            patient_landmarks = {}

            for images, labels, patient_ids, slice_idx, spacing, origin, physical_landmarks, labels_landmarks, slice_axis in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(device), labels.to(device).float()

                labels = labels.view(labels.size(0), -1)
                outputs = model(images)
                
                num_landmarks = 98  
                outputs = outputs.view(outputs.size(0), num_landmarks, 3)
                spacing = spacing.to(outputs.device)
                origin = origin.to(outputs.device)
                labels = labels.view(labels.size(0), num_landmarks, 3)

                # Convert voxel coordinates to physical coordinates
                predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)

                # Only apply reverse rotation if the slice is from rotated data
                if slice_axis == "rotated_axial":
                    predicted_physical = reverse_rotate_landmarks_45(predicted_physical, origin)
                    labels_physical = reverse_rotate_landmarks_45(labels_physical, origin)

                loss = criterion(predicted_physical, labels_physical)
                test_loss += loss.item()

                for i in range(images.size(0)):
                    current_patient_id = patient_ids[i]
                    patient_landmarks[current_patient_id] = predicted_physical[i].cpu().numpy()

            # Save results
            for patient_id, landmarks in patient_landmarks.items():
                patient_fcsv_path = os.path.join(output_folder, f"{patient_id}.fcsv")
                labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data_test/{patient_id}_Fiducial_template_ALL.fcsv"
                print("Patient ID:", patient_id)
                
                header = [
                    "# Markups fiducial file version = 5.6",
                    "# CoordinateSystem = LPS",
                    "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
                ]
                
                labels = []
                if os.path.exists(labels_file):
                    with open(labels_file, 'r') as file:
                        for line in file:
                            if not line.startswith("#") and line.strip():
                                parts = line.strip().split(',')
                                if len(parts) > 11:
                                    labels.append(parts[11])  
                
                with open(patient_fcsv_path, mode="w", newline="") as fcsv_file:
                    writer = csv.writer(fcsv_file, delimiter=',')

                    for line in header:
                        fcsv_file.write(line + "\n")
                    
                    for i, landmark in enumerate(landmarks):
                        label = labels[i] if i < len(labels) else f"Landmark_{i}"
                        writer.writerow([i, *landmark, 0, 0, 0, 0, 1, 1, 0, label, "", ""])

        avg_test_loss = test_loss / len(test_loader)
        print(f"Testing Completed. Test Loss: {avg_test_loss:.4f}")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")

    print(f"Testing complete. Results saved to {results_file} and individual patient files in {output_folder}.")

class MultiDataset(Dataset):
    def __init__(self, folder_path, slice_axis=0, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing .nrrd and .fcsv files.
            slice_axis (int): Axis along which to slice the 3D volume (0 = axial, 1 = coronal, 2 = sagittal).
            transform (callable, optional): Optional transform to apply to the slices.
        """
        self.folder_path = folder_path
        self.slice_axis = slice_axis
        self.transform = transform
        self.pairs = self.match_files(folder_path)
        self.slices_info = self.calculate_slices_info()

    def match_files(self, folder_path):
        """Matches NRRD, segmentation, and FCSV files in the given folder based on patient ID."""
        nrrd_files, seg_files, fcsv_files = {}, {}, {}
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('masked.nrrd'):
                patient_id = file_name.split('_')[0]
                nrrd_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('seg.nrrd'):
                patient_id = file_name.split('_')[0]
                seg_files[patient_id] = os.path.join(folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]
                fcsv_files[patient_id] = os.path.join(folder_path, file_name)
        
        matched_pairs = []
        for patient_id in nrrd_files:
            if patient_id in seg_files and patient_id in fcsv_files:
                matched_pairs.append((patient_id, nrrd_files[patient_id], fcsv_files[patient_id], seg_files[patient_id]))
        
        return matched_pairs

    def calculate_slices_info(self):
        """Calculate the total number of slices and store slice-to-volume mapping."""
        slices_info = []
        for patient_id, nrrd_path, fcsv_path, seg_path in self.pairs:
            volume, _ = nrrd.read(nrrd_path)
            num_slices = volume.shape[self.slice_axis]
            slices_info.extend([(patient_id, nrrd_path, fcsv_path, seg_path, slice_idx) for slice_idx in range(num_slices)])
        return slices_info

    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        patient_id, nrrd_path, fcsv_path, seg_path, slice_idx = self.slices_info[idx]

        # Load masked volume and segmentation volume
        volume, header = nrrd.read(nrrd_path)
        segmentation, _ = nrrd.read(seg_path)
        volume = torch.tensor(volume, dtype=torch.float32)
        segmentation = torch.tensor(segmentation, dtype=torch.float32)
        
        # Extract slices
        if self.slice_axis == 0:
            slice_2d = volume[slice_idx, :, :]
            seg_2d = segmentation[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_2d = volume[:, slice_idx, :]
            seg_2d = segmentation[:, slice_idx, :]
        elif self.slice_axis == 2:
            slice_2d = volume[:, :, slice_idx]
            seg_2d = segmentation[:, :, slice_idx]
        
        # Process landmarks
        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])
        physical_landmarks, labels = process_fcsv_to_tensor(fcsv_path, return_labels=True)
       
        voxel_landmarks = (physical_landmarks - origin) / spacing
        
        # Apply transforms
        if self.transform:
            slice_2d = self.transform(slice_2d)
            seg_2d = self.transform(seg_2d)
        
        return slice_2d, seg_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels



class MultiInputSwinTransformer(nn.Module):
    def __init__(self, num_classes=294):
        super(MultiInputSwinTransformer, self).__init__()

        # Load pretrained Swin Transformer for masked volume
        self.branch1 = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0, in_chans=1)
        self.branch1.head = nn.Identity()  # Remove classification head

        # Load pretrained Swin Transformer for segmentation result
        self.branch2 = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0, in_chans=1)
        self.branch2.head = nn.Identity()  # Remove classification head

        # Feature size from Swin Transformer (for 'swin_base_patch4_window7_224', it's 1024)
        swin_feature_dim = 1024  

        # Fully connected layers after feature fusion
        self.fc = nn.Sequential(
            nn.Linear(swin_feature_dim * 2, 512),  # 1024 features from each branch
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x1, x2):
        # Extract features from both branches
        out1 = self.branch1(x1)  # Masked volume features
        out2 = self.branch2(x2)  # Segmentation features

        # Concatenate features
        merged = torch.cat((out1, out2), dim=1)

        # Fully connected layers
        out = self.fc(merged)
        return out



def train_and_evaluate_model_multi(
    model, train_loader, val_loader, device, epochs=30,
    results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results.txt"
):
    """
    Train and evaluate a Swin Transformer model with segmentation as an additional input.

    Args:
        model: MultiInputSwinTransformer model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to train on (e.g., "cuda" or "cpu").
        epochs: Number of training epochs.
        results_file: File path to save training and validation results.

    Returns:
        best_loss: Best validation loss achieved during training.
        mse: Mean Squared Error for the best validation epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    best_loss = float("inf")  # Track the best validation loss

    with open(results_file, "a") as f:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for images, segmentations, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):

                images, segmentations, labels = images.to(device), segmentations.to(device), labels.to(device).float()
                
                labels = labels.view(labels.size(0), -1)  # Shape: (batch_size, num_landmarks * 3)

                optimizer.zero_grad()
                outputs = model(images, segmentations)  # Multi-input forward pass

                num_landmarks = 98  # Update this if necessary
                outputs = outputs.view(outputs.size(0), num_landmarks, 3)
                labels = labels.view(labels.size(0), num_landmarks, 3)

                spacing = spacing.to(outputs.device)
                origin = origin.to(outputs.device)

                # Convert voxel coordinates to physical space
                predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)

                loss = criterion(predicted_physical, labels_physical)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}\n")

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, segmentations, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in val_loader:
                    images, segmentations, labels = images.to(device), segmentations.to(device), labels.to(device).float()
                    
                    labels = labels.view(labels.size(0), -1)

                    outputs = model(images, segmentations)  # Multi-input forward pass
                    outputs = outputs.view(outputs.size(0), num_landmarks, 3)
                    labels = labels.view(labels.size(0), num_landmarks, 3)

                    spacing = spacing.to(outputs.device)
                    origin = origin.to(outputs.device)

                    predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                    labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)

                    loss = criterion(predicted_physical, labels_physical)
                    val_loss += loss.item()

                    # Collect predictions and ground truth for metrics
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}\n")
            
            # Save the best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/model/swin_multi.pth")

            # Compute and log additional metrics (e.g., MSE)
            all_labels_flat = np.array(all_labels).reshape(-1, 3)
            all_preds_flat = np.array(all_preds).reshape(-1, 3)
            mse = mean_squared_error(all_labels_flat, all_preds_flat)
            print(f"Mean Squared Error (Validation): {mse:.4f}")
            f.write(f"Mean Squared Error (Validation): {mse:.4f}\n")

    return best_loss, mse


def test_for_physical_multi(
    model, 
    test_loader, 
    device, 
    results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results_test_multi.txt", 
    output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/output_test_results_multi"
):
    """
    Test the multi-input model on the test dataset and save results.

    Args:
        model: Trained multi-input model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        results_file: Path to save overall testing results (loss and metrics).
        output_folder: Folder to save individual patient CSV files with predictions.
    """
    os.makedirs(output_folder, exist_ok=True)
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    print("Testing started...")
    test_loss = 0.0

    with open(results_file, "w") as f:
        with torch.no_grad():
            patient_landmarks = {}

            for images, segmentations, labels, patient_ids, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in tqdm(test_loader, desc="Testing"):
                images, segmentations, labels = images.to(device), segmentations.to(device), labels.to(device).float()
                labels = labels.view(labels.size(0), -1)
                
                outputs = model(images, segmentations)  # Multi-input forward pass
                num_landmarks = 98
                outputs = outputs.view(outputs.size(0), num_landmarks, 3)
                labels = labels.view(labels.size(0), num_landmarks, 3)
                spacing = spacing.to(outputs.device)
                origin = origin.to(outputs.device)

                predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)

                loss = criterion(predicted_physical, labels_physical)
                test_loss += loss.item()

                for i in range(images.size(0)):
                    current_patient_id = patient_ids[i]
                    patient_landmarks[current_patient_id] = predicted_physical[i].cpu().numpy()

            for patient_id, landmarks in patient_landmarks.items():
                patient_fcsv_path = os.path.join(output_folder, f"{patient_id}.fcsv")
                labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data/{patient_id}_Fiducial_template_ALL.fcsv"
                print("Patient ID:", patient_id)
                
                header = [
                    "# Markups fiducial file version = 5.6",
                    "# CoordinateSystem = LPS",
                    "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
                ]
                
                labels = []
                if os.path.exists(labels_file):
                    with open(labels_file, 'r') as file:
                        for line in file:
                            if not line.startswith("#") and line.strip():
                                parts = line.strip().split(',')
                                if len(parts) > 11:
                                    labels.append(parts[11])
                
                with open(patient_fcsv_path, mode="w", newline="") as fcsv_file:
                    writer = csv.writer(fcsv_file, delimiter=',')
                    for line in header:
                        fcsv_file.write(line + "\n")
                    
                    for i, landmark in enumerate(landmarks):
                        label = labels[i] if i < len(labels) else f"Landmark_{i}"
                        writer.writerow([i, *landmark, 0, 0, 0, 0, 1, 1, 0, label, "", ""])

        avg_test_loss = test_loss / len(test_loader)
        print(f"Testing Completed. Test Loss: {avg_test_loss:.4f}")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")

    print(f"Testing complete. Results saved to {results_file} and individual patient files in {output_folder}.")


# class NRRDDatasetTest(Dataset):
#     def __init__(self, test_folder, slice_axis=0, transform=None):
#         """
#         Args:
#             test_folder (str): Path to the test folder containing .nrrd files.
#             slice_axis (int): Axis along which to slice the 3D volume (0 = axial, 1 = coronal, 2 = sagittal).
#             transform (callable, optional): Optional transform to apply to the slices.
#         """
#         self.test_folder = test_folder
#         self.slice_axis = slice_axis
#         self.transform = transform
#         self.nrrd_files = self.collect_nrrd_files(test_folder)
#         self.slices_info = self.calculate_slices_info()

#     def collect_nrrd_files(self, folder_path):
#         """Collects NRRD files from the test folder."""
#         nrrd_files = {}
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('masked.nrrd'):
#                 patient_id = file_name.split('_')[0]
#                 nrrd_files[patient_id] = os.path.join(folder_path, file_name)
#         return nrrd_files

#     def calculate_slices_info(self):
#         """Calculate the total number of slices and store slice-to-volume mapping."""
#         slices_info = []
#         for patient_id, nrrd_path in self.nrrd_files.items():
#             volume, header = nrrd.read(nrrd_path)
#             num_slices = volume.shape[self.slice_axis]
#             slices_info.extend([(patient_id, nrrd_path, slice_idx, header) for slice_idx in range(num_slices)])
#         return slices_info

#     def __len__(self):
#         return len(self.slices_info)

#     def __getitem__(self, idx):
#         patient_id, nrrd_path, slice_idx, header = self.slices_info[idx]
#         volume, _ = nrrd.read(nrrd_path)
#         volume = torch.tensor(volume, dtype=torch.float32)

#         if self.slice_axis == 0:
#             slice_2d = volume[slice_idx, :, :]
#         elif self.slice_axis == 1:
#             slice_2d = volume[:, slice_idx, :]
#         elif self.slice_axis == 2:
#             slice_2d = volume[:, :, slice_idx]

#         spacing = np.linalg.norm(header['space directions'], axis=0)
#         origin = np.array(header['space origin'])

#         if self.transform:
#             slice_2d = self.transform(slice_2d)

#         return slice_2d, patient_id, slice_idx, spacing, origin

# def test_for_physical(
#     model, 
#     test_loader, 
#     device, 
#     results_file="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results_test.txt", 
#     output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/output_test_results"
# ):
#     """
#     Test the model on the test dataset and save results.

#     Args:
#         model: Trained model to evaluate.
#         test_loader: DataLoader for test data.
#         device: Device to run the model on (e.g., "cuda" or "cpu").
#         results_file: Path to save overall testing results (loss and metrics).
#         output_folder: Folder to save individual patient CSV files with predictions.
#     """
#     os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

#     model = model.to(device)
#     model.eval()
#     criterion = nn.MSELoss()
#     print(f"Testing started...")
#     test_loss = 0.0

#     with open(results_file, "w") as f:
#         with torch.no_grad():
#             for images, patient_ids, slice_idx, spacing, origin in tqdm(test_loader, desc="Testing"):
#                 images = images.to(device)
#                 spacing = spacing.to(device)
#                 origin = origin.to(device)
                
#                 # Model prediction
#                 outputs = model(images)
                
#                 # Convert voxel coordinates to physical coordinates
#                 predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                
#                 # Compute loss in physical space (dummy comparison to itself for completeness)
#                 loss = criterion(predicted_physical, predicted_physical)
#                 test_loss += loss.item()
                
#                 # Save results per patient
#                 for i in range(images.size(0)):
#                     current_patient_id = patient_ids[i]
#                     patient_fcsv_path = os.path.join(output_folder, f"{current_patient_id}.fcsv")
                    
#                     with open(patient_fcsv_path, mode="w", newline="") as fcsv_file:
#                         writer = csv.writer(fcsv_file, delimiter=',')
#                         writer.writerow(["id", "x", "y", "z"])
                        
#                         for j, landmark in enumerate(predicted_physical[i].cpu().numpy()):
#                             writer.writerow([j, *landmark])
        
#         avg_test_loss = test_loss / len(test_loader)
#         print(f"Testing Completed. Test Loss: {avg_test_loss:.4f}")
#         f.write(f"Test Loss: {avg_test_loss:.4f}\n")
    
#     print(f"Testing complete. Results saved to {results_file} and individual patient files in {output_folder}.")


class LandmarkRefinementModel3D(nn.Module):
    def __init__(self, backbone="r3d_18", num_landmarks=98):
        super(LandmarkRefinementModel3D, self).__init__()

        # Load a 3D ResNet backbone
        if backbone == "r3d_18":
            self.feature_extractor = r3d_18(weights="DEFAULT")
        elif backbone == "mc3_18":
            self.feature_extractor = mc3_18(weights="DEFAULT")
        else:
            raise ValueError("Unsupported 3D backbone")

        # Modify first conv layer to accept 1-channel 3D input (instead of RGB 3-channels)
        self.feature_extractor.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )

        # Modify the final layer to output a fixed feature vector (512D)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, 512)

        # Fully connected layers for landmark refinement
        self.fc1 = nn.Linear(512 + num_landmarks * 3, 256)  # Concatenate 512D features with 294 landmarks
        self.fc2 = nn.Linear(256, num_landmarks * 3)  # Output refined landmarks

        self.relu = nn.ReLU()

    def forward(self, seg_volume, predicted_landmarks):
        batch_size = seg_volume.shape[0]

        # Extract features from 3D segmentation volume
        seg_features = self.feature_extractor(seg_volume)  # Output: [batch, 512]
        
        # Concatenate with predicted landmarks
        x = torch.cat([seg_features, predicted_landmarks.view(batch_size, -1)], dim=1)  # [batch, 512 + 294]
        
        x = self.relu(self.fc1(x))
        refined_landmarks = self.fc2(x)  # Output [batch, 294]
        
        return refined_landmarks.view(batch_size, -1, 3)  # Reshape to [batch, 98, 3]

def train_and_evaluate_refinement_model(
    model, train_loader, val_loader, device, epochs=30,
    results_file="refinement_results.txt",
    refined_pred_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/refined_model_predictions/",
transform=None):
    """
    Train and evaluate the second model that refines the first model's landmark predictions using segmentation volumes.

    Args:
        model: The refinement model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to train on (e.g., "cuda" or "cpu").
        epochs: Number of training epochs.
        results_file: File path to save training and validation results.
        refined_pred_folder: Folder to save refined landmark predictions.

    Returns:
        best_loss: Best validation loss achieved during training.
        mse: Mean Squared Error for the best validation epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.MSELoss()  # MSE for refinement
    
    best_loss = float("inf")
    os.makedirs(refined_pred_folder, exist_ok=True)

    with open(results_file, "a") as f:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_patient_landmarks = {}
            
            for seg_volume, predicted_landmarks, voxel_landmarks, patient_id, spacing, origin, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                if transform:
                    seg_volume = torch.stack([transform(v) for v in seg_volume])
                seg_volume, predicted_landmarks, voxel_landmarks = (
                    seg_volume.to(device),
                    predicted_landmarks.to(device).float(),
                    voxel_landmarks.to(device).float(),
                )
                
                spacing = torch.tensor(spacing, dtype=torch.float32).to(device)
                origin = torch.tensor(origin, dtype=torch.float32).to(device)

                optimizer.zero_grad()
                refined_outputs = model(seg_volume, predicted_landmarks)

                refined_physical = refined_outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                labels_physical = voxel_landmarks * spacing.unsqueeze(1) + origin.unsqueeze(1)

                loss = criterion(refined_physical, labels_physical)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                
                for i in range(seg_volume.shape[0]):
                    patient_name = patient_id[i]
                    train_patient_landmarks[patient_name] = refined_physical[i].detach().cpu().numpy()
            
            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}\n")

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            val_patient_landmarks = {}

            with torch.no_grad():
                for seg_volume, predicted_landmarks, voxel_landmarks, patient_id, spacing, origin, labels in val_loader:
                    if transform:
                        seg_volume = torch.stack([transform(v) for v in seg_volume])
                    seg_volume, predicted_landmarks, voxel_landmarks = (
                        seg_volume.to(device),
                        predicted_landmarks.to(device).float(),
                        voxel_landmarks.to(device).float(),
                    )

                    spacing = torch.tensor(spacing, dtype=torch.float32).to(device)
                    origin = torch.tensor(origin, dtype=torch.float32).to(device)

                    refined_outputs = model(seg_volume, predicted_landmarks)
                    print (f'{patient_id}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                   
                    refined_physical = refined_outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                   
                    labels_physical = voxel_landmarks * spacing.unsqueeze(1) + origin.unsqueeze(1)
                 
                    loss = criterion(refined_physical, labels_physical)
                    val_loss += loss.item()
                    
                    all_preds.extend(refined_outputs.cpu().numpy())
                    all_labels.extend(voxel_landmarks.cpu().numpy())
                    
                    for i in range(seg_volume.shape[0]):
                        patient_name = patient_id[i]
                        val_patient_landmarks[patient_name] = refined_physical[i].detach().cpu().numpy()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}\n")

            # Save the best model and predictions
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/model/seg_model_2.pth")
                save_predictions(refined_pred_folder, train_patient_landmarks)
                save_predictions(refined_pred_folder, val_patient_landmarks)

            all_labels_flat = np.array(all_labels).reshape(-1, 3)
            all_preds_flat = np.array(all_preds).reshape(-1, 3)
            mse = mean_squared_error(all_labels_flat, all_preds_flat)
            print(f"Mean Squared Error (Validation): {mse:.4f}")
            f.write(f"Mean Squared Error (Validation): {mse:.4f}\n")

    return best_loss, mse




class NRRDDatasetRefinement(Dataset):
    def __init__(self, folder_path, first_model_predictions_folder, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing .seg.nrrd and .fcsv files (ground truth landmarks).
            first_model_predictions_folder (str): Path to the folder containing first model's predicted .fcsv files.
            transform (callable, optional): Optional transform to apply to the segmentation volumes.
        """
        self.folder_path = folder_path
        self.first_model_predictions_folder = first_model_predictions_folder
        self.transform = transform

        # Match segmentation and prediction files, and determine if ground truth exists
        self.pairs, self.is_training = self.match_files()

    def match_files(self):
        """Matches segmentation NRRD and FCSV files based on patient ID."""
        seg_files = {}
        gt_fcsv_files = {}  # Ground truth landmarks (for training only)
        pred_fcsv_files = {}  # First model predictions (always used)

        # Load segmentation and ground truth files
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('seg.nrrd'):
                patient_id = file_name.split('_')[0]  # Extract patient ID
                seg_files[patient_id] = os.path.join(self.folder_path, file_name)
            elif file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]
                gt_fcsv_files[patient_id] = os.path.join(self.folder_path, file_name)  # Ground truth

        # Load first model predictions
        for file_name in os.listdir(self.first_model_predictions_folder):
            if file_name.endswith('.fcsv'):
                patient_id = file_name.split('_')[0]
                pred_fcsv_files[patient_id] = os.path.join(self.first_model_predictions_folder, file_name)

        matched_pairs = []
        is_training = False

        for patient_id in seg_files:
            if patient_id in pred_fcsv_files:
                if patient_id in gt_fcsv_files:
                    is_training = True  # Ground truth exists → training mode
                    matched_pairs.append((patient_id, seg_files[patient_id], gt_fcsv_files[patient_id], pred_fcsv_files[patient_id]))
                else:
                    matched_pairs.append((patient_id, seg_files[patient_id], None, pred_fcsv_files[patient_id]))

        return matched_pairs, is_training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        patient_id, seg_path, gt_fcsv_path, pred_fcsv_path = self.pairs[idx]

        # Load segmentation volume
        seg_volume, header = nrrd.read(seg_path)
        seg_volume = torch.tensor(seg_volume, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        spacing = np.linalg.norm(header['space directions'], axis=0)
        origin = np.array(header['space origin'])

        # Load first model predictions
        predicted_landmarks, _ = process_fcsv_to_tensor(pred_fcsv_path, return_labels=True)
        voxel_predicted_landmarks = (predicted_landmarks - origin) / spacing

        # Handle missing ground truth in test mode
        if gt_fcsv_path is not None:
            physical_landmarks, labels = process_fcsv_to_tensor(gt_fcsv_path, return_labels=True)
            voxel_landmarks = (physical_landmarks - origin) / spacing
        else:
            voxel_landmarks = torch.empty((0, 3))  # Empty tensor instead of None
            labels = []  # Empty list instead of None

        # Apply transformation if needed
        if self.transform:
            seg_volume = self.transform(seg_volume)

        return seg_volume, voxel_predicted_landmarks, voxel_landmarks, patient_id, spacing, origin, labels
    
def resize_3d(tensor, target_size=(96, 224, 224)):
    """
    Resizes a 3D volume tensor (C, D, H, W) to target size using trilinear interpolation.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W).
        target_size (tuple): Desired output size (D, H, W).

    Returns:
        torch.Tensor: Resized 3D tensor.
    """
    return F.interpolate(tensor.unsqueeze(0), size=target_size, mode="trilinear", align_corners=False).squeeze(0)

def split_dataset_by_volume_without_2d_slices(dataset, train_volumes, val_volumes):
    """
    Splits the dataset into training and validation subsets based on volume counts.
    
    Args:
        dataset (Dataset): The dataset to be split.
        train_volumes (int): Number of volumes to include in the training set.
        val_volumes (int): Number of volumes to include in the validation set.

    Returns:
        train_set (Subset): Subset of the dataset for training.
        val_set (Subset): Subset of the dataset for validation.
    """
    total_samples = len(dataset)
    total_volumes = train_volumes + val_volumes
    
    if total_samples < total_volumes:
        raise ValueError("Not enough samples in the dataset to split as requested.")
    
    train_indices = list(range(train_volumes))
    val_indices = list(range(train_volumes, train_volumes + val_volumes))
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    return train_set, val_set




def test_refinement_model(
    model, 
    test_loader, 
    device, 
    output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/output_test_results",
    transform=None
):
    """
    Test the landmark refinement model using segmentation volumes and first model predictions.
    
    Args:
        model: Trained refinement model.
        test_loader: DataLoader for test data.
        device: Device for computation (e.g., "cuda" or "cpu").
        output_folder: Folder to save individual patient CSV files with refined predictions.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    
    model.to(device)
    model.eval()
    
    print("Testing started...")
    
    with torch.no_grad():
        patient_landmarks = {}
        
        for seg_volume, predicted_landmarks, _, patient_ids, spacing, origin, _ in tqdm(test_loader, desc="Testing"):
            if transform:
                seg_volume = torch.stack([transform(v) for v in seg_volume])
            seg_volume, predicted_landmarks = (
                seg_volume.to(device),
                predicted_landmarks.to(device).float(),
            )
            
            spacing = spacing.to(device)
            origin = origin.to(device)
            
            # Forward pass: Get refined landmarks
            refined_outputs = model(seg_volume, predicted_landmarks)
            
            # Convert to physical space
            refined_physical = refined_outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
            
            # Store results
            for i in range(seg_volume.size(0)):
                current_patient_id = patient_ids[i]
                patient_landmarks[current_patient_id] = refined_physical[i].cpu().numpy()
            
        # Write each patient's refined physical landmarks to a CSV
        for patient_id, landmarks in patient_landmarks.items():
            patient_fcsv_path = os.path.join(output_folder, f"{patient_id}.fcsv")
            labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data_test/{patient_id}_Fiducial_template_ALL.fcsv"
            print(f"Saving refined landmarks for {patient_id}")
            
            # Define the header lines
            header = [
                "# Markups fiducial file version = 5.6",
                "# CoordinateSystem = LPS",
                "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
            ]
            
            # Read label information from the provided labels file
            labels = []
            if os.path.exists(labels_file):
                with open(labels_file, 'r') as file:
                    for line in file:
                        if not line.startswith("#") and line.strip():  # Ignore comments and empty lines
                            parts = line.strip().split(',')
                            if len(parts) > 11:  # Ensure enough columns exist to fetch the label
                                labels.append(parts[11])  # The label is at index 11
            
            # Write to the .fcsv file
            with open(patient_fcsv_path, mode="w", newline="") as fcsv_file:
                writer = csv.writer(fcsv_file, delimiter=',')
                for line in header:
                    fcsv_file.write(line + "\n")
                
                for i, landmark in enumerate(landmarks):
                    label = labels[i] if i < len(labels) else f"{i}"  # Assign default if missing
                    writer.writerow([i, *landmark, 0, 0, 0, 0, 1, 1, 0, label, "", ""])
    
    print("Testing completed.")
