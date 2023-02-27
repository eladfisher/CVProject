"""Define your architecture here."""
from xcpetion import build_xception_backbone
import torch
from models import SimpleNet

import torch.nn.functional as F

from torch import nn




def my_new_competition_model():
    """Return an Xception-Based network.

    build our competetive nn based on the Xception NN
    """
    Xception_model_Fake = build_xception_backbone(pretrained=True)
    Xception_model_Fake.fc = nn.Sequential(nn.Dropout(0.1),
                                      nn.Linear(2048, 1024),
                                      nn.ReLU(),
                                      nn.Linear(1024, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 2)

    )

    return Xception_model_Fake

def my_competition_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = my_new_competition_model()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/competition_model.pt')['model'])
    return model
