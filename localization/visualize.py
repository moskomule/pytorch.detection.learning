import os
from PIL import Image
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from torchvision import transforms
import torch
from torch.autograd import Variable

from data_storage import id_class

totensor = transforms.Compose([transforms.ToTensor()])
topilimg = transforms.Compose([transforms.ToPILImage()])
cuda = torch.cuda.is_available()


def topk_2d(input, k):
    """
    get 2d matrix's top-k largest elements' positions
    """
    w, h = input.size()
    input_ = input.view(-1)
    _, idx = torch.sort(input_, 0, descending=True)
    return [(idx[i] // w, idx[i] % w) for i in range(k)]


def find_cls(input):
    input = input.data.max(0)[1]
    return input.sum()


def get_bbox_points(loc, grid_x, grid_y, grid_size, im_w, im_h, image_size):
    x, y, w, h = loc[:, grid_x, grid_y]
    # grid center
    gc_x, gc_y = (grid_x + 0.5) * grid_size, (grid_y + 0.5) * grid_size
    # bounding box's center
    x, y = gc_x + (x * grid_size / 2), gc_y + (y * grid_size / 2)
    x, y = x * im_w / image_size, y * im_h / image_size
    # bounding box's width and height
    w, h = w * im_w / 2, h * im_h / 2
    x_0 = int(max(x - w - 1, 0))
    y_0 = int(max(y - h - 1, 0))
    x_1 = int(min(x + w, im_w) - 1)
    y_1 = int(min(y + h, im_h) - 1)
    return x_0, x_1, y_0, y_1


def create_bounding_box(image_path, image_size, model, *, dpi=120, topk=4, grid_num=5):
    model.eval()

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
    grids = topk_2d(output_cnf, topk)
    grid_size = image_size // grid_num

    # load image for matplotlib
    image = mpimg.imread(image_path)
    fig = Figure(figsize=(im_h/dpi, im_w/dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.imshow(image)

    for grid_x, grid_y in grids:
        x_0, x_1, y_0, y_1 = get_bbox_points(output_loc.data, grid_x, grid_y, grid_size,
                                             im_w, im_h, image_size)
        ax.plot([y_0, y_0, y_1, y_1, y_0], [x_0, x_1, x_1, x_0, x_0])
        ax.text(y_0, x_0, f"{id_class[find_cls(output_cls[:, grid_x, grid_y])]}",
                bbox={'alpha': 0.5})
        ax.axis('off')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    # numpy array -> PIL data -> tensor
    tensor = totensor(Image.fromarray(image))
    # image is numpy array and tensor is tensor
    return image, tensor


class PlotLoss:
    def __init__(self, vis, win=None, opts=None):
        self.vis = vis
        self.win = win
        self.opts = opts
        if self.win is None:
            self.win = self.opts.get("title")
        self.__iteration = 0
        self.container = [[], []]

    def append(self, data):
        self.__iteration += 1
        if isinstance(data, list):
            assert len(data) <= 2, "data length should be 1 or 2"
            self.container[0].append(data[0])
            self.container[1].append(data[1])
            X = np.array([range(self.__iteration), range(self.__iteration)]).T
            Y = np.array(self.container).T
        else:
            self.container[0].append(data)
            X = np.array([range(self.__iteration)])
            Y = np.array(self.container[0])

        self.vis.line(X=X, Y=Y, win=self.win, opts=self.opts)


class ShowSample:
    def __init__(self, model, vis, image_path, image_size, grid_num, win=None, opts=None):
        self.model = model
        assert os.path.exists(image_path)
        self.image_path = image_path
        self.image_size = image_size
        self.grid_num = grid_num
        self.vis = vis
        self.win = win
        self.opts = opts

    def show(self):
        _, tensor = create_bounding_box(self.image_path, self.image_size,
                                        self.model, grid_num=self.grid_num)
        self.vis.image(img=tensor, win=self.win, opts=self.opts)

if __name__ == '__main__':
    """
    this is kind of test code
    """
    from model import YOLOlike
    import matplotlib.pyplot as plt
    model = YOLOlike()
    o, l = create_bounding_box("2012_000004.jpg", 150, "sample", model)
    plt.imshow(o)
    plt.axis("off")
    plt.show()
