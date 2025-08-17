import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    """Base CNN class with common functionality"""
    
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
    
    def num_flat_features(self, x):
        """Calculate number of features after flattening"""
        size = x.size()[1:]  # Exclude batch dimension
        return size.numel()
    
    def feature2logit(self, x):
        """Map features to logits using final classifier layer"""
        return self.fc2(x)


class CifarCNN(BaseCNN):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        
        # Parameter grouping for federated learning
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
            'conv3.weight', 'conv3.bias', 'fc1.weight', 'fc1.bias'
        ]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias']

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y


class AnimalCNN(BaseCNN):
    def __init__(self, num_classes=10):
        super(AnimalCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # After 3 pooling ops: 32->16->8->4
        self.fc2 = nn.Linear(128, num_classes)
        
        # Parameter grouping for federated learning
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
            'conv3.weight', 'conv3.bias', 'fc1.weight', 'fc1.bias'
        ]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias']

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y


class Cifar100CNN(BaseCNN):
    def __init__(self, num_classes=100):
        super(Cifar100CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)  # After 4 pooling ops: 32->16->8->4->2
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Parameter grouping for federated learning
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
            'conv3.weight', 'conv3.bias', 'conv4.weight', 'conv4.bias',
            'fc1.weight', 'fc1.bias'
        ]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias']

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return x, y


class CNN_FMNIST(BaseCNN):
    def __init__(self, num_classes=10):
        super(CNN_FMNIST, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        
        # Parameter grouping for federated learning
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
            'fc1.weight', 'fc1.bias'
        ]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias']

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y


class Discriminator(nn.Module):
    """Simple discriminator for adversarial training"""
    
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Autoencoder(nn.Module):
    """Simple autoencoder for dimensionality reduction"""
    
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)
    
    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        return self.decoder(encoded)