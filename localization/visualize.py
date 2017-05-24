import os
from PIL import Image

from torchvision import transforms
import torch
from torch.autograd import Variable

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


def create_bounding_box(image_name, image_size, data_root, model):
    model.eval()

    image_path = os.path.join(data_root, image_name)
    image = Image.open(image_path)
    im_h, im_w = image.size
    origin = totensor(image)

    # unsqueeze for make batch_size = 1
    input = totensor(image.resize([image_size, image_size])).unsqueeze(0)
    if cuda:
        input = input.cuda()
    output_loc, output_cls = model(Variable(input))
    if cuda:
        output_loc = output_loc.cpu()
        output_cls = output_cls.cpu()
    # squeeze because batch is 1
    output_cls.data.squeeze_(0), output_loc.data.squeeze_(0)
    # get high confidence grids
    idx = topk_2d(output_loc.data[4, :, :], 4)
    bdbox_t = [output_loc.data[:4, w, h] for w, h in idx]
    for loc in bdbox_t:
        x, y, w, h = loc * torch.FloatTensor([im_w, im_h, im_w/2, im_h/2])
        x_0 = int(max(x - w - 1, 0))
        y_0 = int(max(y - h - 1, 0))
        x_1 = int(min(x + w, im_w) - 1)
        y_1 = int(min(y + h, im_h) - 1)
        origin = _plot_rectangle(origin, [x_0, x_1, y_0, y_1])
    return origin
