import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class_mapping = {"roads":1,"fields":0}


class Field_or_Road(Dataset):
    """
    Field or Road dataset loader, the class of an image is determined by the folder it is in ("fields" or "roads")

    Parameters
    ----------
    image_dir : string,
        Path of the directory of images with the name starting by the id of the image
    transform : albumentation.compose
        A composition of the different data augmentation with their probabilities during the training and data processing  
    """
    def __init__(self, image_dir, transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir,self.images[index])
        
        ground_truth = class_mapping[Path(img_path).name.split("_")[0]]
        #need numerical values for classifier : 1 = roads & 0 = fields
        
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform is not None:
            augmentations = self.transform(image = image)
            image = augmentations["image"]
        return image,ground_truth