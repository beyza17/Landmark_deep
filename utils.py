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
            for images, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
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
                
                print(f"Labels shape (before reshape): {labels.shape}")
                labels = labels.view(labels.size(0), num_landmarks, 3)
                print(f"Labels shape (after reshape): {labels.shape}")
                print(f"Spacing shape: {spacing.unsqueeze(1).shape}")
                print(f"Origin shape: {origin.unsqueeze(1).shape}")


                predicted_physical = outputs * spacing.unsqueeze(1) + origin.unsqueeze(1)
                labels_physical = labels * spacing.unsqueeze(1) + origin.unsqueeze(1)
                
                # Print voxel and physical coordinates for the first patient along with patient_id
                if epoch == 0:  # Only print during the first epoch
                    first_patient_id = patient_id[0]  # Patient ID of the first patient in the batch
                    patient_voxel_coords = labels[0].view(-1, 3)[:5]  # First 5 landmarks in voxel space
                    patient_physical_coords = labels_physical[0].view(-1, 3)[:5]  # Corresponding physical coordinates

                    print("First Patient before prediction (Physical Coordinates):")
                    print(physical_landmarks.cpu().numpy())
                    print(f"\nPatient ID: {first_patient_id}")
                    print("First Patient (Voxel Coordinates):")
                    print(patient_voxel_coords.cpu().numpy())
                    print("First Patient after prediction (Physical Coordinates):")
                    print(patient_physical_coords.cpu().numpy())
                
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
                for images, labels, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks in val_loader:
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
                torch.save(model.state_dict(), "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/model/swin_best_with_physical_loss_2.pth")

            
            # Compute and log additional metrics (e.g., MSE)
            all_labels_flat = np.array(all_labels).reshape(-1, 3)  # [batch_size * num_landmarks, 3]
            all_preds_flat = np.array(all_preds).reshape(-1, 3)    # [batch_size * num_landmarks, 3]
            mse = mean_squared_error(all_labels_flat, all_preds_flat)
            print(f"Mean Squared Error (Validation): {mse:.4f}")
            f.write(f"Mean Squared Error (Validation): {mse:.4f}\n")

    return best_loss, mse

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




def test_for_physical(
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
                labels_file = f"/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data_test/{patient_id}_Fiducial_template_ALL.fcsv"
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
