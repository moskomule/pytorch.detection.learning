import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from numba import jit

cuda = torch.cuda.is_available()
WEIGHT = torch.ones(21)/20
WEIGHT[0] = 0  # "nothing"
if cuda:
    WEIGHT = WEIGHT.cuda()
nll_loss_2d = NLLLoss2d(weight=WEIGHT)


@jit
def iou(pred, gt):
    """
    iou
    todo: make it faster using Cython etc.
    """
    px_0 = (pred[0] - pred[2])*10
    px_1 = (pred[0] + pred[2])*10
    py_0 = (pred[1] - pred[3])*10
    py_1 = (pred[1] + pred[3])*10

    gx_0 = (gt[0] - gt[2])*10
    gx_1 = (gt[0] + gt[2])*10
    gy_0 = (gt[1] - gt[3])*10
    gy_1 = (gt[1] + gt[3])*10

    x_0 = max(px_0, gx_0)
    x_1 = min(px_1, gx_1)
    y_0 = max(py_0, gy_0)
    y_1 = min(py_1, gy_1)

    iarea = max(x_1 - x_0, 0) * max(y_1 - y_0, 0) + 1
    parea = 4 * (pred[2]+1) * (pred[3]+1)
    garea = 4 * (gt[2]+1) * (gt[3]+1)

    return iarea/(parea+garea-iarea)


def filter_logsoftmax(x):
    """
    softmax along filters
    """

    assert x.dim() == 4, "dimension of input must be 4"
    _, f, _, _ = x.size()
    nom = torch.exp(x)
    din = torch.sum(nom, 1).repeat(1, f, 1, 1)
    return x-torch.log(din)


def cls_loss_func(output, target, mask):
    """
    class loss function
    """
    # pow2 distance from output to target
    dist = torch.pow(output - target, 2)
    # sum along classes and flatten
    flat = torch.sum(dist, 1).view(-1)
    # if no class for a grid, ignore it (mask `flat`)
    return torch.sum(flat * mask.float().view(-1), 0)/torch.sum(mask.float())


def loc_loss_func(output, target, mask):
    """
    location loss function
    """
    # pow 2 distance from output tor target
    # todo scale output because w,h is half
    dist = torch.pow(output - target, 2)
    # sum along classes and flatten
    flat = torch.sum(dist, 1).view(-1)
    return torch.sum(flat*mask.float().view(-1), 0)/torch.sum(mask.float())


def cnf_loss_func(output_loc, output_cnf, target_loc, mask, gamma):
    """
    confidence loss function
    """
    # confidence is
    def _cnf_loss(b, w, h):
        p_box = output_loc[b, :, w, h]
        g_box = target_loc[b, :, w, h]
        return (output_cnf[b, 0, w, h] - iou(p_box, g_box)) ** 2

    # inv mask: 0 if there exists an object in a cell otherwise 1
    inv_mask = 1 - mask

    obj = (_cnf_loss(b, w, h) for b, w, h in torch.nonzero(mask))
    no_obj = (torch.pow(output_cnf[b, 0, w, h], 2) for b, w, h
              in torch.nonzero(inv_mask))

    return sum(obj)/torch.sum(mask.float()) \
           + gamma * sum(no_obj)/torch.sum(inv_mask.float())


def yololike_loss(output, target, alpha, beta, gamma):
    output_loc, output_cnf, output_cls = output

    target_loc = target[:, :4, :, :]
    target_cls = target[:, 4:, :, :]
    # mask: 1 if there exists an object in a cell otherwise 0
    mask = Variable(target_cls.data.sum(1).gt(0).squeeze(1),
                    requires_grad=False)
    if cuda:
        mask = mask.cuda()

    loc_loss = loc_loss_func(output_loc, target_loc, mask)
    cnf_loss = cnf_loss_func(output_loc, output_cnf, target_loc,
                             mask.data, gamma)
    cls_loss = cls_loss_func(output_cls, target_cls, mask)

    total = loc_loss + (alpha * cls_loss) + (beta * cnf_loss)

    return total, loc_loss, cnf_loss, cls_loss
