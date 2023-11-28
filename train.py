import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils.CNN_model import VGG16,VGG16_test,SimpleCNN

from utils.model_utils import (
     load_checkpoint,
     save_checkpoint,
     get_loaders,
     check_accuracy,
 )

# Hyperparameters

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 2
NUM_WORKERS = 4
IMAGE_HEIGHT =  224
IMAGE_WIDTH = 336
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/train"
VAL_IMG_DIR = "dataset/val"
TEST_IMG_DIR = "dataset_original/test_images"
MODEL_NAME = "model_name"


def train(loader, model, optimizer, loss_fn):
    """Train the model for one epoch using pytorch library

    Returns:
        int : Average training loss of the epoch
    """
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = nn.functional.one_hot(targets,num_classes=2).to(device = DEVICE) #hot encoding for CrossEntropyLoss

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets.float())
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())
    return loss.item()

def validation(loader,model,optimizer,loss_fn,best_loss):
    """Validate the model after the training of an epoch using pytorch library
    

    Returns:
        int : Average loss of the validation set
        int : Updated value of the best loss of the current training
    """
    with torch.no_grad():
            correct = 0
            total = 0
            total_val_loss =0
            for images, labels in loader:
                images = images.to(device= DEVICE)
                label_hot = nn.functional.one_hot(labels,num_classes=2).to(device = DEVICE)
                labels = labels.to(device = DEVICE)
                
                outputs = model(images)

                #accuracy %
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                #loss value
                val_loss = loss_fn(outputs,label_hot.float())
                total_val_loss += val_loss.item() * len(labels)

                del images, labels, outputs

            avg_val_loss = total_val_loss / total
            print('Val loss for this epoch = {}'.format(avg_val_loss))
            print('Accuracy of the network on the {} validation images: {} %'.format(18, 100 * correct / total)) 
            checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            }
            if avg_val_loss < best_loss:
                print('New best loss !')
                save_checkpoint(checkpoint,filename=MODEL_NAME+"_best")
                best_loss = avg_val_loss

            #save_checkpoint(checkpoint,filename=MODEL_NAME + "_last.pt") #commented taking too long each epochs
    return avg_val_loss, best_loss

def main():
    """
    Run the model training with basic parameters and evaluate the model thanks to validation set
    Save loss history into a plot

    """

    #img pre-processing (resize + augment + to tensor)
    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit = 15, p = 1.0),
            A.HorizontalFlip(p= 0.5),
            #A.VerticalFlip(p=0.1),
            #A.GaussNoise(p=0.3),
            #A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1 ,1, 1],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    #no augment for val but still normalise and resize
    val_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],       #mean=[0.485, 0.456, 0.406], standard normalisation parameters for ImagNet
                std=[1 ,1, 1],              #std=[0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    

    #multiple architecture tested for CNN and already made architecture VGG16, needed simple model for simple dataset to avoid overfitting
    model = SimpleCNN(IMAGE_HEIGHT,IMAGE_WIDTH).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,1]).to(device = DEVICE)) #use weight to balance training here or while spliting data
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE) #weight decay didn't help improving results


    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        VAL_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    #load a model to keep training or eval
    if LOAD_MODEL:
        load_checkpoint(torch.load(MODEL_NAME +".pt"), model)
        check_accuracy(val_loader,model,device=DEVICE)
        

    best_loss_val = float('inf') #set to high value
    last_train_loss =0
    val_loss_history = []
    train_loss_history =[]
    for epoch in range(NUM_EPOCHS):
        #Training
        model.train()
        last_train_loss = train(train_loader, model, optimizer, loss_fn)
        train_loss_history.append(last_train_loss)

        # Validation
        model.eval()
        last_loss_val, best_loss_val = validation(val_loader,model,optimizer,loss_fn,best_loss_val) #val is also unbalanced so use of same CrossEntropy weight
        val_loss_history.append(last_loss_val)
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        }
    save_checkpoint(checkpoint,filename=MODEL_NAME + "_last.pt")
        
    #Show loss history
    epoch_list = np.linspace(1,NUM_EPOCHS,NUM_EPOCHS)
    plt.figure()
    plt.plot(epoch_list,val_loss_history,color="r",label="Val Loss")
    plt.plot(epoch_list,train_loss_history,color="g",label="Train Loss")
    plt.title("Val loss vs Train loss")
    plt.savefig("Train loss vs Val loss")


if __name__ == "__main__":
    
    main()
