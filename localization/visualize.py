import os
from PIL import Image
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from torchvision import transforms
import torch
from torch.nn.functional import softmax
from torch.autograd import Variable

from data_storage import id_class

totensor = transforms.Compose([transforms.ToTensor()])
topilimg = transforms.Compose([transforms.ToPILImage()])
cuda = torch.cuda.is_available()


def topk_2d(input, k):
    w, h = input.size()
    input_ = input.view(-1)
    _, idx = torch.sort(input_, 0, descending=True)
    l = []
    for i in range(k):
        l.append([idx[i] // w, idx[i] % w])

    return l


def find_cls(c_tensor):
    tensor = softmax(c_tensor).data.max(0)[1]
    return tensor.sum()


def _plot_rectangle(i_tensor, cood):
    x_0, x_1, y_0, y_1 = cood
    try:
        i_tensor[:, x_0:x_1, y_0] = 1
        i_tensor[:, x_0:x_1, y_1] = 1
        i_tensor[:, x_0, y_0:y_1] = 1
        i_tensor[:, x_1, y_0:y_1] = 1
    except ValueError:
        # some time rectangle is line
        pass
    return i_tensor


def create_bounding_box(image_name, image_size, data_root, model, *, dpi=120):
    model.eval()

    image_path = os.path.join(data_root, image_name)
    image = Image.open(image_path)
    im_h, im_w = image.size

    # unsqueeze for make batch_size = 1
    input = totensor(image.resize([image_size, image_size])).unsqueeze(0)
    if cuda:
        input = input.cuda()
    output_loc, output_cnf, output_cls = model(Variable(input))
    if cuda:
        output_loc = output_loc.cpu()
        output_cnf = output_cnf.cpu()
        output_cls = output_cls.cpu()
    # squeeze because batch is 1
    output_cls.data.squeeze_(0), output_loc.data.squeeze_(0)
    output_cnf = output_cnf.data[0, 0, :, :]
    # get high confidence grids
    idx = topk_2d(output_cnf, 8)

    bdbox_t = [output_loc.data[:, w, h] for w, h in idx]
    cls_list = [find_cls(output_cls[:, w, h]) for w, h in idx]
    cnf_list = [output_cnf[w, h] for w, h in idx]

    image = mpimg.imread(image_path)
    fig = Figure(figsize=(im_h/dpi, im_w/dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.imshow(image)

    for loc, cnf, cls in zip(bdbox_t, cnf_list, cls_list):
        x, y, w, h = loc * torch.FloatTensor([im_w, im_h, im_w/2, im_h/2])
        x_0 = int(max(x - w - 1, 0))
        y_0 = int(max(y - h - 1, 0))
        x_1 = int(min(x + w, im_w) - 1)
        y_1 = int(min(y + h, im_h) - 1)
        ax.plot([y_0, y_0, y_1, y_1, y_0], [x_0, x_1, x_1, x_0, x_0])
        ax.text(y_0, x_0, f"{id_class[cls]}", bbox={'alpha': 0.5})
        ax.axis('off')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    tensor = totensor(Image.fromarray(image))
    # image is numpy array and tensor is tensor
    return image, tensor


if __name__ == '__main__':
    """
    this is kind of test code
    """
    from yolo_like import YOLOlike
    import matplotlib.pyplot as plt
    model = YOLOlike()
    o, l = create_bounding_box("2012_000004.jpg", 150, "sample", model)
    plt.imshow(o)
    plt.axis("off")
    plt.show()
