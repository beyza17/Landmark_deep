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
from utils import NRRDDatasetDynamicSlices,NRRDDatasetMultiAxis,NRRDDatasetMultiAxis_test,NRRDDatasetMultiAxis_diag1_rotated,test_for_physical_rotated,process_fcsv_to_tensor,split_dataset_by_volume,train_and_evaluate_model, test_for_physical,test_for_voxel

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
folder_path = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/data_test_another_dataset"

# Create the dataset
# test_set = NRRDDatasetDynamicSlices(folder_path=folder_path, slice_axis=0,transform=transform)
test_set = NRRDDatasetMultiAxis_test(folder_path=folder_path, transform=transform, test_mode=True)

# Access a sample
slice_2d, voxel_landmarks, patient_id, slice_idx, spacing, origin, physical_landmarks, labels_landmarks = test_set[0]

# Print the results
print("Patient ID:", patient_id)
print("Slice shape:", slice_2d.shape)  # Should be [1, 224, 224]
# print("Voxel landmarks:", voxel_landmarks.shape)



# Create data loaders for the subsets
batch_size = 32

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print(f"Train set size: {len(test_set)} slices")

model = create_model('resnet101', pretrained=True, num_classes=294, in_chans=1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to(device)

state_dict = torch.load('/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/model/resenet101_5.pth', map_location=device)

# Load the weights with strict=False to skip incompatible keys
model.load_state_dict(state_dict, strict=False)



results_file = "/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/result_deep/swin_results_physical_test_6.txt"
output_folder="/work/shared/ngmm/scripts/Beyza_Zayim/Beyza/predictions6"
if os.path.exists(results_file):
    os.remove(results_file)  
#test_for_physical(model, test_loader, device,results_file=results_file, output_folder=output_folder) ##NRRDDatasetMultiAxis
test_for_physical(model, test_loader, device, output_folder=output_folder) #NRRDDatasetMultiAxis_test

