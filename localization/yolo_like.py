import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from preprocessor import VocDataSet, get_annotaions
from functions import yololike_loss

cuda = torch.cuda.is_available()


def variable(tensor):
    if cuda:
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)

resnet = models.resnet50()
reshead = torch.nn.Sequential(*list(resnet.children())[:-2])


class YOLOlike(nn.Module):
    def __init__(self):
        super(YOLOlike, self).__init__()
        self.base_model = reshead
        self.conv = torch.nn.Conv2d(2048, 625, kernel_size=1)

    def forward(self, x):
        x = self.base_model(x)
        return self.conv(x)


def train(model, optimizer, train_loader, epochs):
    for epoch in epochs:
        for data, target, _, _ in train_loader:
            data, target = variable(data), variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = yololike_loss(output, target)
            loss.backward()


if __name__ == '__main__':
    yolo_like = YOLOlike()
    if cuda:
        yolo_like.cuda()
    optimizer = torch.optim.Adam(yolo_like.parameters())
    train_a, test_a = get_annotaions("sample")
    train_loader = DataLoader(VocDataSet(train_a, image_dir="sample"))
    test_loader = DataLoader(VocDataSet(test_a, image_dir="sample"))

