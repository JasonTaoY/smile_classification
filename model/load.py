import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


def initialize_model(num_classes=2, pretrained=True, device='cuda'):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, loss, optimizer


