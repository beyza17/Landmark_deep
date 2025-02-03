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
from utils import NRRDDatasetDynamicSlices,process_fcsv_to_tensor,split_dataset_by_volume,train_and_evaluate_model

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA on device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device = torch.device("cpu")
    print("Using CPU for computations.")

# Define transform pipeline
transform = T.Compose([
    T.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),  # Ensure slices have channel dim [1, H, W]
    T.Resize((224, 224)),  # Resize to (224, 224)
])

# Path to the folder containing NRRD and FCSV files
folder_path = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_original/data"

# Create the dataset
dataset = NRRDDatasetDynamicSlices(folder_path=folder_path, slice_axis=0,transform=transform)


# Access a sample
slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks = dataset[0]

# Print the results
print("Patient ID:", patient_id)
print("Slice shape:", slice_2d.shape)  # Should be [1, 224, 224]
print("Voxel landmarks:", voxel_landmarks.shape)



train_volumes = 10
val_volumes = 2


# Split the dataset based on volumes
train_set, val_set = split_dataset_by_volume(dataset, train_volumes, val_volumes)

# Create data loaders for the subsets
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# Print dataset lengths
print(f"Train set size: {len(train_set)} slices")
print(f"Validation set size: {len(val_set)} slices")


model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=294,in_chans=1)
model = model.to(device)

results_file = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results_physical_3.txt"
if os.path.exists(results_file):
    os.remove(results_file)  
train_and_evaluate_model(model, train_loader, val_loader, device, epochs=80, results_file=results_file)
    
