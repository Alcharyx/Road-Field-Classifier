import torch
import torchvision
from utils.dataset import Field_or_Road
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint"):
    """
    Save the current model state into a file

    """
    print("=> Saving checkpoint")
    torch.save(state,filename +".pt")

def load_checkpoint(checkpoint, model):
    """
    Load a checkpoint file
    
    """
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
        train_dir,
        val_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers = 4,
        pin_memory = True):
    """
    Generate training and validation data loader for the Field or Road dataset structure

    Parameters
    ----------

    train_dir : string
        Path of the directory of training images with the name starting by the id of the image ex :"615_XXXX.jpg" to reference to the correct mask
    val_dir : string
        Path of the directory of validation images with the name starting by the id of the image ex :"615_XXXX.jpg" to reference to the correct mask
    batch_size : int
        size of a batch during  training
    train_transform : albumentation.compose
        A composition of the different data augmentation and processing for training
    val_transform : albumentation.compose
        A composition of the different data processing for validation (resize and normalize only) 
    num_workers : int, optional
        Number of worker to load the data
    pin_memory : bool, optional
        Enable pin memory for more efficient data transfer host <=> device
    
    """
    train_ds = Field_or_Road(image_dir= train_dir,
                                transform= train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers= num_workers,
        pin_memory= pin_memory,
        shuffle= True,
    )

    val_ds = Field_or_Road(
        image_dir= val_dir,
        transform= val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size= batch_size,
        num_workers= num_workers,
        pin_memory= pin_memory,
        shuffle= False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, plot=False, device="cuda"):
    """
    Get accuracy on the loader dataset
    If plot enable plot all the images of the dataset with Ground Truth and Prediction label.
    If model is correct the title is Green else the title is red
    
    """
    model.eval() #eval mode

    with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(device= device)
                labels = labels.to(device = device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if plot == True:
                    for index in range(len(labels)):
                        plt.imshow(images[0].permute(1,2,0))
                        if predicted[index] == labels[index]:
                            color = "green"
                        else:
                            color = "red"
                        plt.title('Classification : {} | Ground Truth : {}'.format(predicted[index],labels[index]),color=color)
                        plt.show()
                del images, labels, outputs
            print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total)) 
    model.train()

