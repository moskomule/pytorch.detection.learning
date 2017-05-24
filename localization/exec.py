import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

import visdom

from preprocessor import VocDataSet, get_annotations
from yolo_like import YOLOlike, train, _debug
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

    tot_loss, loc_loss, cls_loss = [], [], []
    for i in range(1000):
        print(f"epoch {i}")
        if arg.debug:
            a, b, c = _debug(yololike, optimizer, train_loader)
        else:
            a, b, c = train(yololike, optimizer, train_loader)
        print("visualize sample")
        o, l = create_bounding_box("2012_000004.jpg", 150, "sample", yololike)
        o2, l2 = create_bounding_box("8435.jpg", 150, "sample", yololike)
        l_id = [id_class[i] for i in l]
        l2_id = [id_class[i] for i in l2]
        print(l_id)
        print(l2_id)
        tot_loss.append(a)
        loc_loss.append(b)
        cls_loss.append(c)
        viz.line(X=np.arange(i+1), Y=np.array(tot_loss), win="loss", opts=dict(title="total loss"))
        viz.line(X=np.arange(i+1), Y=np.array(loc_loss), win="loc_loss", opts=dict(title="localization loss"))
        viz.line(X=np.arange(i+1), Y=np.array(cls_loss), win="cls_loss", opts=dict(title="classification loss"))
        viz.image(img=o, win="test_image")
        viz.image(img=o2, win="test_image")
