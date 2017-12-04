import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Variable


cuda = torch.cuda.is_available()
resnet = models.resnet34(pretrained=True)
# use layers before average pooling
reshead = torch.nn.Sequential(*list(resnet.children())[:-2])


def variable(tensor):
    if cuda:
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


class YOLORes(nn.Module):
    def __init__(self, image_size):
        super(YOLORes, self).__init__()
        self.features = nn.Sequential(*[m if not isinstance(m, nn.modules.activation.ReLU)
                                        else nn.LeakyReLU(inplace=True) for m in reshead])
        _, c, h, w = self.features(Variable(torch.randn(1, 3, image_size, image_size))).size()
        self.grid_size = h
        self.__features_output_size = c * h * w
        self.fc = nn.Sequential(
            nn.Linear(self.__features_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # class = 20
        self.cls = nn.Linear(1024, 20 * self.grid_size * self.grid_size)
        # loc = 4 points
        self.loc = nn.Linear(1024, 4 * self.grid_size * self.grid_size)
        # confidence
        self.cnf = nn.Linear(1024, 1 * self.grid_size * self.grid_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.__features_output_size)
        x = self.fc(x)
        loc_output = self.loc(x).view(-1, 4, 5, 5)
        cnf_output = self.cnf(x).view(-1, 1, 5, 5)
        cls_output = self.cls(x).view(-1, 20, 5, 5)
        return F.sigmoid(loc_output), F.sigmoid(cnf_output), F.sigmoid(cls_output)


class Yolov1(nn.Module):
    def __init__(self, image_size):
        super(Yolov1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            )

        _, c, h, w = self.features(Variable(torch.randn(1, 3, image_size, image_size))).size()
        self.grid_size = h
        self.__features_output_size = c * h * w
        self.fc = nn.Sequential(
            nn.Linear(self.__features_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            )

        # class = 20
        self.cls = nn.Linear(1024, 20 * self.grid_size * self.grid_size)
        # loc = 4 points
        self.loc = nn.Linear(1024, 4 * self.grid_size * self.grid_size)
        # confidence
        self.cnf = nn.Linear(1024, 1 * self.grid_size * self.grid_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.__features_output_size)
        x = self.fc(x)
        loc_output = self.loc(x).view(-1, 4, self.grid_size, self.grid_size)
        cnf_output = self.cnf(x).view(-1, 1, self.grid_size, self.grid_size)
        cls_output = self.cls(x).view(-1, 20, self.grid_size, self.grid_size)
        return F.sigmoid(loc_output), F.sigmoid(cnf_output), F.sigmoid(cls_output)


if __name__ == '__main__':
    pass
