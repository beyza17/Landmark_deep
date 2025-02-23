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
from utils import resize_3d,NRRDDatasetRefinement,train_and_evaluate_refinement_model,LandmarkRefinementModel3D,split_dataset_by_volume_without_2d_slices
import random
import torch.nn.functional as F
import torchio as tio


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA on device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device = torch.device("cpu")
    print("Using CPU for computations.")


def add_gaussian_noise(volume, mean=0.0, std=0.05):
    """Add random Gaussian noise to the 3D volume."""
    noise = torch.randn_like(volume) * std + mean
    return volume + noise

def adjust_intensity(volume, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    """Apply random intensity scaling and shifting."""
    scale = random.uniform(*scale_range)
    shift = random.uniform(*shift_range)
    return volume * scale + shift



def gaussian_blur_3d(volume, kernel_size=3):
    """Apply slight Gaussian blur using 3D convolution."""
    padding = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, kernel_size, kernel_size) / (kernel_size ** 3)
    weight = weight.to(volume.device)

    # Ensure volume is at least 4D: [C, D, H, W]
    if volume.dim() == 3:  # If input is [D, H, W], add channel dim
        volume = volume.unsqueeze(0)  # [1, D, H, W]

    if volume.dim() == 4:  # If input is [C, D, H, W], add batch dim
        volume = volume.unsqueeze(0)  # [1, C, D, H, W]

    # Now volume should be [1, 1, D, H, W]
    assert volume.dim() == 5, f"Unexpected input shape {volume.shape}"

    # Apply 3D convolution
    blurred = F.conv3d(volume, weight, padding=padding)

    return blurred.squeeze(0)  # Remove batch dimension to return [C, D, H, W]




transform = T.Compose([
    T.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),  # Convert to tensor
    T.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 3 else x),  # Ensure shape (1, D, H, W)
    T.Lambda(lambda x: resize_3d(x, target_size=(96, 224, 224))),  # Resize using trilinear interpolation
    T.Lambda(lambda x: add_gaussian_noise(x, std=0.03)),  # Add slight Gaussian noise
    T.Lambda(lambda x: adjust_intensity(x, scale_range=(0.95, 1.05), shift_range=(-0.05, 0.05))),  # Intensity variation
    T.Lambda(lambda x: gaussian_blur_3d(x, kernel_size=3)),  # Apply fixed blur
    T.Lambda(lambda x: tio.ZNormalization()(x)),  # **Histogram normalization (zero-mean, unit variance)**
    T.Lambda(lambda x: tio.RandomAffine(degrees=15)(x)),  # **Random 15-degree rotation**
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize voxel intensities
])





# Path to the folder containing NRRD and FCSV files
folder_path = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data"
first_model_predictions = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/predictions_training_copy"

dataset = NRRDDatasetRefinement(folder_path=folder_path, first_model_predictions_folder=first_model_predictions, transform=transform)


# Access a sample
seg_volume, predicted_landmarks, voxel_landmarks, patient_id, spacing, origin, labels = dataset[0]

print(f"Total number of 3D volumes: {len(dataset)}")

train_volumes = 8
val_volumes = 2


# Split the dataset based on volumes
train_set, val_set = split_dataset_by_volume_without_2d_slices(dataset, train_volumes, val_volumes)

# Create data loaders for the subsets
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# Print dataset lengths
print(f"Train set size: {len(train_set)} volumes")
print(f"Validation set size: {len(val_set)} volumes")


model = LandmarkRefinementModel3D(backbone='mc3_18', num_landmarks=98).to(device)
model = model.to(device)


results_file = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/seg_results_physical_1.txt"
refined_pred_folder = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/predictions_training_refined2/"
if os.path.exists(results_file):
    os.remove(results_file)  
train_and_evaluate_refinement_model(model, train_loader, val_loader, device, epochs=1000, results_file=results_file,refined_pred_folder=refined_pred_folder)