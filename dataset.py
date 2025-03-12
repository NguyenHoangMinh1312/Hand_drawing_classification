"""Get the hand drawing images and its label"""
from torch.utils.data import Dataset
import os
import numpy as np
import re
import cv2
import torch

class HandDrawingDataset(Dataset):
    def __init__(self, root, train = True, ratio = 0.8, mean = None, std = None, image_size = 224):
        self.image_paths = []
        self.npy_arrays = []    #store the numpy arrays
        self.labels = []
        self.mean, self.std = mean, std  
        self.root = root
        self.image_size = image_size
        self.categories = [self.extract_category(name) for name in os.listdir(root)]

        for i, f in enumerate(os.listdir(root)):  
            data = np.load(os.path.join(root, f))
            data = np.array(data[:int(len(data) * 0.1)])    #only use 10% of the data
            self.npy_arrays.append(data)

            tmp_image_paths = []
            tmp_labels = []
            for j in range(len(data)):
                tmp_image_paths.append({
                    "npy_array_id": i,   #the id of the numpy array which contains this image
                    "image_id": j        #the index of the image in the numpy array
                })
                tmp_labels.append(i)
            
            split_idx = int(len(data) * ratio)
            if train:
                self.image_paths.extend(tmp_image_paths[:split_idx])
                self.labels.extend(tmp_labels[:split_idx])
            else:
                self.image_paths.extend(tmp_image_paths[split_idx:])
                self.labels.extend(tmp_labels[split_idx:])

    #extract the category name from the numpy folder
    def extract_category(self, file_name):
        match = re.search(r"bitmap_(\w+)", file_name)
        return match.group(1) if match else None 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        npy_array_id = self.image_paths[idx]["npy_array_id"]
        image_id = self.image_paths[idx]["image_id"]
        image = self.npy_arrays[npy_array_id][image_id].reshape(28, -1)

        #preprocess the image
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        label = self.labels[idx]
        label = torch.tensor(label, dtype = torch.int64)
        return image, label

    #get the mean and std of the train set
    def get_mean_std(self):
        pixel_sum = 0
        pixel_suqare_sum = 0
        pixel_num = 0

        for image_path in self.image_paths:
            npy_array_id = image_path["npy_array_id"]
            image_id = image_path["image_id"]
            image = self.npy_arrays[npy_array_id][image_id].reshape(28, -1)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image / 255
            
            pixel_sum += np.sum(image)
            pixel_suqare_sum += np.sum(image ** 2)
            pixel_num += self.image_size ** 2
       
        mean = pixel_sum / pixel_num
        std = np.sqrt(pixel_suqare_sum / pixel_num - mean ** 2)
        return mean, std
     