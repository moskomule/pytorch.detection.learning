import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F

from functions import yololike_loss

from tqdm import tqdm

cuda = torch.cuda.is_available()
resnet = models.resnet34(pretrained=True)
# we use layers before average pooling
reshead = torch.nn.Sequential(*list(resnet.children())[:-2])


def variable(tensor):
    if cuda:
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


class YOLOlike(nn.Module):
    def __init__(self):
        super(YOLOlike, self).__init__()
        self.base_model = reshead
        # class = 20
        self.conv_cls = nn.Conv2d(512, 20, kernel_size=1)
        # loc = 4 points
        self.conv_loc = nn.Conv2d(512, 4, kernel_size=1)
        # confidence
        self.conv_cnf = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = self.base_model(x)
        loc_output = self.conv_loc(x)
        cnf_output = self.conv_cnf(x)
        cls_output = self.conv_cls(x)
        return F.sigmoid(loc_output), F.sigmoid(cnf_output), F.sigmoid(cls_output)


class YOLOlikeFC(nn.Module):
    def __init__(self):
        super(YOLOlikeFC, self).__init__()
        self.base_model = reshead
        # class = 20
        self.cls = nn.Linear(512, 20)
        # loc = 4 points
        self.loc = nn.Linear(512, 4)
        # confidence
        self.cnf = nn.Linear(512, 1)

    def forward(self, x):
        x = self.base_model(x).view(-1, 512)
        loc_output = self.loc(x).view(-1, 4, 5, 5)
        cnf_output = self.cnf(x).view(-1, 1, 5, 5)
        cls_output = self.cls(x).view(-1, 20, 5, 5)
        return F.sigmoid(loc_output), F.sigmoid(cnf_output), F.sigmoid(cls_output)


def train(model, optimizer, train_loader, *, debug=False, verbose=0,
          alpha=0.2, beta=0.2, gamma=0.5):

    model.train()
    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_cnf_loss = 0
    epoch_loc_loss = 0
    for data, target, _, _ in tqdm(train_loader, desc="train"):
        data, target = variable(data), variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss, loc_loss, cnf_loss, cls_loss = yololike_loss(output, target, alpha, beta, gamma)
        epoch_loss += loss.data[0] / len(train_loader)
        epoch_loc_loss += loc_loss.data[0] / len(train_loader)
        epoch_cnf_loss += cnf_loss.data[0] / len(train_loader)
        epoch_cls_loss += cls_loss.data[0] / len(train_loader)
        loss.backward()
        optimizer.step()

        if debug:
            print(f"DEBUG MODE LOSS{epoch_loss}")
            break

        elif verbose > 0:
            print(f"loss:{loss.data[0]}")
            print(f"tlocloss:{loc_loss.data[0]}")
            print(f"cnfloss:{cnf_loss.data[0]}")
            print(f"clsloss:{cls_loss.data[0]}")

    return epoch_loss, epoch_loc_loss, epoch_cnf_loss, epoch_cls_loss


def _debug(model, optimizer, train_loader):
    return train(model, optimizer, train_loader, debug=True, verbose=1)


def test(model, test_loader, alpha=0.2, beta=0.2, gamma=0.5):
    model.eval()

    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_cnf_loss = 0
    epoch_loc_loss = 0
    for data, target, size, scale in tqdm(test_loader, desc="test"):
        data, target = variable(data), variable(target)
        output = model(data)
        loss, loc_loss, cnf_loss, cls_loss = yololike_loss(output, target, alpha, beta, gamma)
        epoch_loss += loss.data[0] / len(test_loader)
        epoch_loc_loss += loc_loss.data[0] / len(test_loader)
        epoch_cnf_loss += cnf_loss.data[0] / len(test_loader)
        epoch_cls_loss += cls_loss.data[0] / len(test_loader)
    return epoch_loss, epoch_loc_loss, epoch_cnf_loss, epoch_cls_loss


if __name__ == '__main__':
    pass
