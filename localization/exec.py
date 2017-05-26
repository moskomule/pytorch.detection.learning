import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

import visdom

from preprocessor import VocDataSet, get_annotations
from yolo_like import YOLOlike, train, _debug, test
from visualize import create_bounding_box
from data_storage import id_class

cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--debug", action="store_true")
    arg = parse.parse_args()

    VOC_BASE = os.path.join("..", "data", "VOC2012")
    IMAGE_SIZE = 150

    viz = visdom.Visdom(port=6006)
    yololike = YOLOlike()
    if cuda:
        yololike.cuda()
    optimizer = torch.optim.Adam(yololike.parameters())
    train_a, test_a = get_annotations(VOC_BASE)
    train_loader = DataLoader(VocDataSet(train_a, dir=VOC_BASE), batch_size=64, num_workers=4)
    test_loader = DataLoader(VocDataSet(test_a, dir=VOC_BASE))

    tot_loss, loc_loss, cnf_loss, cls_loss = [], [], [], []
    _tot_loss, _loc_loss, _cnf_loss, _cls_loss = [], [], [], []
    for i in range(1000):
        print(f"epoch {i}")
        if arg.debug:
            a, b, c, d = _debug(yololike, optimizer, train_loader)
            e, f, g, h = _debug(yololike, optimizer, train_loader)
        else:
            a, b, c, d = train(yololike, optimizer, train_loader, verbose=0)
            e, f, g, h = test(yololike, test_loader)
        print("visualize sample")
        _, o = create_bounding_box("2012_000004.jpg", 150, "sample", yololike)
        _, o2 = create_bounding_box("8435.jpg", 150, "sample", yololike)
        tot_loss.append(a)
        loc_loss.append(b)
        cnf_loss.append(c)
        cls_loss.append(d)
        _tot_loss.append(e)
        _loc_loss.append(f)
        _cnf_loss.append(g)
        _cls_loss.append(h)

        X = np.array([range(i+1), range(i+1)]).T
        viz.line(X=X, Y=np.array([tot_loss, _tot_loss]).T, win="loss", opts=dict(title="total loss"))
        viz.line(X=X, Y=np.array([loc_loss, _loc_loss]).T, win="loc_loss", opts=dict(title="localization loss"))
        viz.line(X=X, Y=np.array([cnf_loss, _cnf_loss]).T, win="cnf_loss", opts=dict(title="confidence loss"))
        viz.line(X=X, Y=np.array([cls_loss, _cls_loss]).T, win="cls_loss", opts=dict(title="classification loss"))
        viz.image(img=o, win="test_image")
        viz.image(img=o2, win="test_image2")
