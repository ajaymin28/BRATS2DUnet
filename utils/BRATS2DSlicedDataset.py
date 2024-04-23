import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import gc

class BRATS2DSlicedDataset(Dataset):
    def __init__(self, dataset_slices, channels_to_use=0, wholeTumor=True, transforms= None):
        """
        channel options :  ch0: FLAIR, ch1: T1 , ch2: T1c, ch3: T2
        wholeTumor : Predict whole tumor. If you want to predict tumor sections, then  pass false. 
        L0 = 0      # Background
        L1 = 50     # Necrotic and Non-enhancing Tumor
        L2 = 100    # Edema
        L3 = 150    # Enhancing Tumor
        """
        self.dataset = dataset_slices
        self.channels_to_use = channels_to_use
        self.wholeTumor = wholeTumor
        self.transforms = transforms
        self.remove_slices_without_annotations()

    def remove_slices_without_annotations(self):
        refined_dataset = []
        # for data in self.dataset_batch:
        data = self.dataset

        data['image'] = data['image'][self.channels_to_use,:,:,:] # 0 for using only flair channel 
        label = data['label'][0]
        if self.wholeTumor:  # cobine 
            label[label > 0] = 1.0
            data['label'] = label
        slice_indexes = []
        for lb_idx in range(label.shape[2]):
            if np.any(label[:, :, lb_idx] > 0): # check if any labels are there or not.
                slice_indexes.append(lb_idx)

        # height, width, slices = label.shape
        for idx, label_idx in enumerate(slice_indexes):
            sliced_label = label[:,:,label_idx]
            sliced_image = data['image'][:,:,label_idx]
            refined_dataset.append((sliced_image, sliced_label))
        self.dataset = refined_dataset
        del refined_dataset
        gc.collect()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = torch.Tensor(image).unsqueeze(0)  # Adding channel dimension
        label = torch.Tensor(label).unsqueeze(0)
        return image, label