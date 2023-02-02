"""Define your architecture here."""
import torch
from models import SimpleNet

import torch.nn.functional as F

from torch import nn


class MyModel(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(24 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        fully_connected_third_out = F.relu(self.fc3(fully_connected_second_out))
        two_way_output = self.fc4(fully_connected_third_out)
        return two_way_output


def my_new_competition_model():
    return MyModel()

def my_competition_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = MyModel()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/competition_model.pt')['model'])
    return model
