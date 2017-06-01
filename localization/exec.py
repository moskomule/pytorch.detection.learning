import os
import argparse

import torch
from torch.utils.data import DataLoader

import visdom
from tqdm import tqdm

from preprocessor import VocDataSet, get_annotations
from model import YOLORes, Yolov1, variable
from functions import yololike_loss
from visualize import PlotLoss, ShowSample

cuda = torch.cuda.is_available()


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
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--debug", action="store_true")
    parse.add_argument("-r", "--res", action="store_false")
    parse.add_argument("-e", "--epoch", type=int, default=1000)
    arg = parse.parse_args()

    VOC_BASE = os.path.join("..", "data", "VOC2012")
    IMAGE_SIZE = 300
    BATCH_SIZE = 64

    model = Yolov1(IMAGE_SIZE) if arg.res else YOLORes(IMAGE_SIZE)

    GRID_NUM = model.grid_size

    viz = visdom.Visdom(port=6006)

    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_a, test_a = get_annotations(VOC_BASE)
    train_loader = DataLoader(VocDataSet(train_a, image_size=IMAGE_SIZE, grid_num=GRID_NUM, dir=VOC_BASE),
                              batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(VocDataSet(test_a, IMAGE_SIZE, grid_num=GRID_NUM, dir=VOC_BASE),
                             batch_size=BATCH_SIZE, num_workers=4)

    ttl_loss = PlotLoss(viz, opts=dict(title="total loss"))
    loc_loss = PlotLoss(viz, opts=dict(title="localization loss"))
    cls_loss = PlotLoss(viz, opts=dict(title="classification loss"))
    cnf_loss = PlotLoss(viz, opts=dict(title="confidence loss"))
    sample1 = ShowSample(model, viz, "sample/2012_000004.jpg", IMAGE_SIZE, win="test1", grid_num=GRID_NUM)
    sample2 = ShowSample(model, viz, "sample/8435.jpg", IMAGE_SIZE, win="test2", grid_num=GRID_NUM)

    print("-"*10)
    print(model)
    print("-"*10)

    for i in tqdm(range(arg.epoch), desc="total"):
        if arg.debug:
            a, b, c, d = _debug(model, optimizer, train_loader)
            e, f, g, h = _debug(model, optimizer, train_loader)
        else:
            # loss function
            # total = loc_loss + (alpha * cls_loss) + (beta * (obj_cnf_loss + gamma * noobj_cnf_loss))
            # alpha=0.2, beta=0.2, gamma=0.5
            a, b, c, d = train(model, optimizer, train_loader, verbose=0, alpha=0.5)
            e, f, g, h = test(model, test_loader, alpha=0.5)

        ttl_loss.append([a, e])
        loc_loss.append([b, f])
        cnf_loss.append([c, g])
        cls_loss.append([d, h])
        sample1.show()
        sample2.show()
