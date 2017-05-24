import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import visdom

from preprocessor import VocDataSet, get_annotations
from yolo_like import YOLOlike, train, _debug
from visualize import create_bounding_box

cuda = torch.cuda.is_available()

if __name__ == '__main__':
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
        a, b, c = _debug(yololike, optimizer, train_loader)
        print("visualize sample")
        o = create_bounding_box("2012_000004.jpg", 150, "sample", yololike)
        tot_loss.append(a)
        loc_loss.append(b)
        cls_loss.append(c)
        viz.line(X=np.arange(i+1), Y=np.array(tot_loss), win="loss")
        viz.line(X=np.arange(i+1), Y=np.array(loc_loss), win="loc_loss")
        viz.line(X=np.arange(i+1), Y=np.array(cls_loss), win="cls_loss")
        viz.image(img=o, win="test_image")
