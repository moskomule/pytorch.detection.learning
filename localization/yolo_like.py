import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F

from functions import yololike_loss, count_correct

cuda = torch.cuda.is_available()
resnet = models.resnet50(pretrained=True)
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
        self.conv_cls = nn.Conv2d(2048, 20, kernel_size=1)
        # loc = 4 points
        self.conv_loc = nn.Conv2d(2048, 4, kernel_size=1)
        # confidence
        self.conv_cnf = nn.Conv2d(2048, 1, kernel_size=1)

    def forward(self, x):
        x = self.base_model(x)
        loc_output = F.sigmoid(self.conv_loc(x))
        cnf_output = F.sigmoid(self.conv_cnf(x))
        return loc_output, cnf_output, self.conv_cls(x)


def train(model, optimizer, train_loader, *, debug=False, verbose=0):

    model.train()
    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_cnf_loss = 0
    epoch_loc_loss = 0
    for i, (data, target, _, _) in enumerate(train_loader):
        data, target = variable(data), variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss, loc_loss, cnf_loss, cls_loss = yololike_loss(output, target)
        epoch_loss += loss.data[0] / len(train_loader)
        epoch_loc_loss += loc_loss.data[0] / len(train_loader)
        epoch_cnf_loss += cnf_loss.data[0] / len(train_loader)
        epoch_cls_loss += cls_loss.data[0] / len(train_loader)
        loss.backward()
        optimizer.step()

        if debug:
            print(f"DEBUG MODE LOSS{epoch_loss}")
            break

        elif i % 10 == 0 and verbose > 0:
            print(f"iteration:[{i}] loss:{loss.data[0]}")
            print(f"iteration:[{i}] locloss:{loc_loss.data[0]}")
            print(f"iteration:[{i}] cnfloss:{cnf_loss.data[0]}")
            print(f"iteration:[{i}] clsloss:{cls_loss.data[0]}")

    return epoch_loss, epoch_loc_loss, epoch_cnf_loss, epoch_cls_loss


def _debug(model, optimizer, train_loader):
    return train(model, optimizer, train_loader, debug=True, verbose=1)


def test(model, test_loader):
    model.eval()

    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_cnf_loss = 0
    epoch_loc_loss = 0
    for data, target, size, scale in test_loader:
        data, target = variable(data), variable(target)
        output = model(data)
        loss, loc_loss, cnf_loss, cls_loss = yololike_loss(output, target)
        epoch_loss += loss.data[0] / len(test_loader)
        epoch_loc_loss += loc_loss.data[0] / len(test_loader)
        epoch_cnf_loss += cnf_loss.data[0] / len(test_loader)
        epoch_cls_loss += cls_loss.data[0] / len(test_loader)
    return epoch_loss, epoch_loc_loss, epoch_cnf_loss, epoch_cls_loss


if __name__ == '__main__':
    pass
