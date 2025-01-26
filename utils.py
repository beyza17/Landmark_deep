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
def process_fcsv_to_tensor(fcsv_file):
    """
    Parses an FCSV file to extract physical coordinates (x, y, z) as a PyTorch tensor.
    
    Args:
        fcsv_file (str): Path to the FCSV file.
        
    Returns:
        torch.Tensor: Tensor of shape (num_landmarks, 3) containing physical coordinates.
    """
    # Open and read the FCSV file
    with open(fcsv_file, 'r') as f:
        lines = f.readlines()

    # List to store coordinates
    coordinates = []

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

    # Convert to PyTorch tensor
    return torch.tensor(coordinates, dtype=torch.float32)
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
            if file_name.endswith('.nrrd'):
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
        physical_landmarks = process_fcsv_to_tensor(fcsv_path)
        voxel_landmarks = (physical_landmarks - origin) / spacing

        # Apply transformations
        if self.transform:
            slice_2d = self.transform(slice_2d)

        


        return slice_2d, voxel_landmarks, patient_id, slice_idx



def split_dataset_by_volume(dataset, train_volumes, val_volumes, test_volumes):
    """
    Splits a dataset into train, validation, and test sets based on volumes.
    
    Args:
        dataset (Dataset): The full dataset.
        train_volumes (int): Number of volumes for training.
        val_volumes (int): Number of volumes for validation.
        test_volumes (int): Number of volumes for testing.
    
    Returns:
        tuple: Train, validation, and test datasets.
    """
    # Extract unique volumes from the dataset
    unique_volumes = list(set(info[0] for info in dataset.slices_info))  # patient_id is at index 0
    random.shuffle(unique_volumes)  # Shuffle to randomize the split

    # Ensure the specified number of volumes matches the dataset
    assert train_volumes + val_volumes + test_volumes == len(unique_volumes), \
        "The sum of train, val, and test volumes must equal the total number of unique volumes."

    # Split the volumes
    train_volume_ids = unique_volumes[:train_volumes]
    val_volume_ids = unique_volumes[train_volumes:train_volumes + val_volumes]
    test_volume_ids = unique_volumes[train_volumes + val_volumes:]

    # Filter slices by volume IDs
    train_indices = [idx for idx, info in enumerate(dataset.slices_info) if info[0] in train_volume_ids]
    val_indices = [idx for idx, info in enumerate(dataset.slices_info) if info[0] in val_volume_ids]
    test_indices = [idx for idx, info in enumerate(dataset.slices_info) if info[0] in test_volume_ids]

    # Create subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set

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
            for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(device), labels.to(device).float() 
                
                # Flatten labels to match outputs
                labels = labels.view(labels.size(0), -1)  # Shape: (batch_size, num_landmarks * 3)
                
                optimizer.zero_grad()
                outputs = model(images)  # Shape: (batch_size, num_landmarks * 3)
                loss = criterion(outputs, labels)  # MSE Loss
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
                for images, labels, _, _  in val_loader:
                    images, labels = images.to(device), labels.to(device).float() 
                    
                    # Flatten labels to match outputs
                    labels = labels.view(labels.size(0), -1)  # Shape: (batch_size, num_landmarks * 3)
                    
                    outputs = model(images)  # Shape: (batch_size, num_landmarks * 3)
                    loss = criterion(outputs, labels)
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
                torch.save(model.state_dict(), "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/model/swin_best.pth")

            # Compute and log additional metrics (e.g., MSE)
            mse = mean_squared_error(np.array(all_labels), np.array(all_preds))
            print(f"Mean Squared Error (Validation): {mse:.4f}")
            f.write(f"Mean Squared Error (Validation): {mse:.4f}\n")

    return best_loss, mse
