import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF


class VGG16_CNN_block(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,in_channels,out_channels):
        super(VGG16_CNN_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class VGG16_FCN_block(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,in_channels,out_channels):
        super(VGG16_FCN_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.fc(x)


class VGG16(nn.Module):
    """Simple VGG16 architecture

    """
    def __init__(self, num_classes=2, in_channels = 3, out_channels =1, features_cnn = [64,64,128,128,256,256,256,512,512,512,512,512,512], when_pool = [1,3,6,9,12],features_fcn = [25088,4096,4096], dropout_p = 0.5):
        super(VGG16,self).__init__()
        self.CNN_layers = nn.ModuleList()
        self.FCN_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_p)
        # CNN part
        for layer_number in range(0,len(features_cnn)):
            self.CNN_layers.append(VGG16_CNN_block(in_channels,features_cnn[layer_number]))
            if layer_number in when_pool:
                self.CNN_layers.append(self.pool)
            in_channels = features_cnn[layer_number]

        # FCN part
        in_channels = features_fcn[0]
        for layer_number in range(1,len(features_fcn)):
            self.FCN_layers.append(VGG16_FCN_block(in_channels,features_fcn[layer_number]))
            self.FCN_layers.append(self.dropout)
            in_channels = features_fcn[layer_number]
        self.FCN_layers.append(nn.Linear(in_channels,num_classes))
        #add softmax
    
    def forward(self,x):
        for layer in self.CNN_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.FCN_layers:
            x = layer(x)
        return x

class SimpleCNN(nn.Module):
    """SimpleCNN architecture, simple model for simple dataset

    """
    def __init__(self, img_width,img_height,num_classes =2):
        super(SimpleCNN, self).__init__()
        # CNN part
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # FCN part
        size_fc1 = 32 * (img_height // 4) * (img_width //4) #automatically adjust if changing img siz. Needs to be change if model architecture modified
        self.fc1 = nn.Linear(size_fc1, 128) 
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, num_classes) 
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.flatten(x)
        x = self.dropout1(self.relu4(self.fc1(x)))
        x = self.fc3(x)
        return nn.functional.softmax(x,dim=1)