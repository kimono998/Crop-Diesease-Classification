import numpy as np
import torch
import torch
from torch import nn
import torch.optim as optim

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class SimpleModel(nn.Module):
    #1 hidden layer neural network, no convolution
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleModel, self).__init__()

        self.fc_block1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))

        self.fc_block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes))

        self.apply(init_weights)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block1(x)
        return self.fc_block2(x)

class BasicCNN(nn.Module):
    #1 CNN layer network
    def __init__(self, input_dim, num_filter, k_size, pool_size, dense_size,num_classes, stride, img_x, img_y):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, num_filter, kernel_size=k_size, stride=stride, padding='same'),
                                  nn.BatchNorm2d(num_filter),
                                  nn.ReLU(),
                                  nn.Conv2d(num_filter, num_filter, kernel_size=k_size, stride=stride, padding='same'),
                                  nn.BatchNorm2d(num_filter),
                                  nn.ReLU(),
                                  nn.MaxPool2d(pool_size)
        )

        self.fc_block = nn.Sequential(nn.Linear((img_x*img_y*num_filter)//(pool_size**2), dense_size),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.2),
                                      nn.Linear(dense_size, num_classes)
        )

        self.apply(init_weights)

    def forward(self, x):

        x = self.conv1(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc_block(x)

        return x

class StackedCNN(nn.Module):
    #A model with stacked convolutional layers.
    def __init__(self, input_dim, num_filter, k_size, pool_size, dense_size,num_classes, stride, img_x, img_y):
        super(StackedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
            nn.Conv2d(num_filter, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter*2),
            nn.ReLU(),
            nn.Conv2d(num_filter*2, num_filter * 2, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter * 2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.fc_block = nn.Sequential(
            nn.Linear((img_x*img_y*num_filter*2)//pool_size**4, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, num_classes)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)

        return x

class StackedCNN2(nn.Module):
    #A model with stacked convolutional layers.
    def __init__(self, input_dim, num_filter, k_size, pool_size, dense_size,num_classes, stride, img_x, img_y):
        super(StackedCNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
            nn.Conv2d(num_filter, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter*2),
            nn.ReLU(),
            nn.Conv2d(num_filter*2, num_filter * 2, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter * 2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter*4, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter*4),
            nn.ReLU(),
            nn.Conv2d(num_filter*4, num_filter * 4, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter * 4),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.fc_block1 = nn.Sequential(
            nn.Linear((img_x*img_y*num_filter*4)//pool_size**6, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, dense_size)
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(dense_size, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, num_classes)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block1(x)
        x = self.fc_block2(x)

        return x




class StackedNoNorm(nn.Module):
    #A model with stacked convolutional layers.
    def __init__(self, input_dim, num_filter, k_size, pool_size, dense_size,num_classes, stride, img_x, img_y):
        super(StackedNoNorm, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.fc_block = nn.Sequential(
            nn.Linear((img_x*img_y*num_filter)//pool_size**6, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)

        return x
class DeepNetwork(nn.Module):
    #A model with stacked convolutional layers.
    def __init__(self, input_dim, num_filter, k_size, pool_size, dense_size,num_classes, stride, img_x, img_y):
        super(DeepNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter*2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.fc_block = nn.Sequential(
            nn.Linear((img_x*img_y*num_filter)//pool_size**6, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(dense_size, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, num_classes)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        x = self.fc_block2(x)

        return x


class DeepNetwork2(nn.Module):
    #more shallow replication of VGG
    def __init__(self, input_dim, num_filter, k_size, pool_size, dense_size,num_classes, stride, img_x, img_y):
        super(DeepNetwork2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.Conv2d(num_filter, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.Conv2d(num_filter*2, num_filter*2, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter*4, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.Conv2d(num_filter * 4, num_filter * 4, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_filter*4, num_filter*8, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.Conv2d(num_filter * 8, num_filter * 8, kernel_size=k_size, stride=stride, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.fc_block = nn.Sequential(
            nn.Linear((img_x*img_y*num_filter*8)//(pool_size**8), dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(dense_size, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, num_classes)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        x = self.fc_block2(x)

        return x


class DeepNetwork3(nn.Module):
    #more shallow replication of VGG
    def __init__(self, input_dim, num_filter, k_size, pool_size, dense_size,num_classes, stride, img_x, img_y):
        super(DeepNetwork3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, num_filter, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter*2, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter*2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter*4, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter*4),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_filter*4, num_filter*8, kernel_size=k_size, stride=stride, padding='same'),
            nn.BatchNorm2d(num_filter*8),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.fc_block = nn.Sequential(
            nn.Linear((img_x*img_y*num_filter*8)//(pool_size**8), dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(dense_size, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, num_classes)
        )

        self.apply(init_weights)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        x = self.fc_block2(x)

        return x
