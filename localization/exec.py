import os
import argparse

import torch
from torch.utils.data import DataLoader

import visdom
from tqdm import tqdm

from preprocessor import VocDataSet, get_annotations
from yolo_like import YOLOlike, YOLOlikeFC, train, _debug, test
from visualize import PlotLoss, ShowSample

cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--debug", action="store_true")
    parse.add_argument("-e", "--epoch", type=int, default=1000)
    arg = parse.parse_args()

    VOC_BASE = os.path.join("..", "data", "VOC2012")
    IMAGE_SIZE = 150
    GRID_NUM = 5
    BATCH_SIZE = 64

    viz = visdom.Visdom(port=6006)
    yololike = YOLOlikeFC()
    # yololike = YOLOlike()
    if cuda:
        yololike.cuda()
    optimizer = torch.optim.Adam(yololike.parameters(), lr=5e-4)
    train_a, test_a = get_annotations(VOC_BASE)
    train_loader = DataLoader(VocDataSet(train_a, image_size=IMAGE_SIZE, grid_num=GRID_NUM, dir=VOC_BASE),
                              batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(VocDataSet(test_a, IMAGE_SIZE, grid_num=GRID_NUM, dir=VOC_BASE),
                             batch_size=BATCH_SIZE, num_workers=4)

    ttl_loss = PlotLoss(viz, opts=dict(title="total loss"))
    loc_loss = PlotLoss(viz, opts=dict(title="localization loss"))
    cls_loss = PlotLoss(viz, opts=dict(title="classification loss"))
    cnf_loss = PlotLoss(viz, opts=dict(title="confidence loss"))
    sample1 = ShowSample(yololike, viz, "sample/2012_000004.jpg", IMAGE_SIZE, win="test1", grid_num=GRID_NUM)
    sample2 = ShowSample(yololike, viz, "sample/8435.jpg", IMAGE_SIZE, win="test2", grid_num=GRID_NUM)

    for i in tqdm(range(arg.epoch), desc="total"):
        if arg.debug:
            a, b, c, d = _debug(yololike, optimizer, train_loader)
            e, f, g, h = _debug(yololike, optimizer, train_loader)
        else:
            a, b, c, d = train(yololike, optimizer, train_loader, verbose=0)
            e, f, g, h = test(yololike, test_loader)

        ttl_loss.append([a, e])
        loc_loss.append([b, f])
        cnf_loss.append([c, g])
        cls_loss.append([d, h])
        sample1.show()
        sample2.show()
