import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.LeakyReLU(0.99)
        self.dropout1 = nn.Dropout(p=0.1)  # drop out
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.LeakyReLU(0.001)
        self.dropout2 = nn.Dropout(p=0.1)  
        self.fc3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout after activation
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout after activation
        x = self.fc3(x)
        return x