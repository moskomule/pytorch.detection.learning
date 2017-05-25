import torch
import torch.nn.functional as F
from torch.nn import NLLLoss2d

cuda = torch.cuda.is_available()
WEIGHT = torch.ones(21)
WEIGHT[0] = 0  # "nothing"
if cuda:
    WEIGHT = WEIGHT.cuda()
nll_loss_2d = NLLLoss2d(weight=WEIGHT)

def iou(pred, gt):
    """
    >>> a=iou([0.4, 0.5, 0.1, 0.2], [0.5, 0.5, 0.1, 0.2])
    >>> (a-1/3) < 1e-2
    True
    >>> iou([0.4, 0.5, 0.1, 0.2], [0.1, 0.5, 0.1, 0.2])
    0.0
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


def filter_softmax(x):
    """
    softmax along filters
    """

    assert x.dim() == 4, "dimension of input must be 4"
    _, f, _, _ = x.size()
    nom = torch.exp(x)
    din = torch.sum(nom, 1).repeat(1, f, 1, 1)
    return torch.log(nom/din)


def cls_loss_func(output, target):
    """
    class loss function
    >>> from torch.autograd import Variable as V
    >>> o, t = V(torch.randn(1,21,3,3)), V(torch.zeros(1,3,3).long())
    >>> t[0,2,1], t[0,2,2] = 1, 10
    >>> cls_loss_func(o, t).data.size()
    torch.Size([1])
    """
    # class loss function
    output = filter_softmax(output)
    return nll_loss_2d(output, target)


def loc_loss_func(output, target, mask):
    """
    location loss function
    >>> from torch.autograd import Variable as V
    >>> o,t,m = V(torch.zeros(1,3,3,3)),V(torch.zeros(1,3,3,3)),torch.zeros(1,3,3)
    >>> o[0,2,0,0] = 1; m[0,0,0] = 1
    >>> loc_loss_func(o, t, m) # distance should be ...
    1.0
    """
    dist = torch.pow(output-target, 2)
    flat = torch.sum(dist, 1).view(-1)
    return torch.sum(flat*mask.float().view(-1), 0)


def cnf_loss_func(output_loc, output_cnf, output_cls, target_loc, target_cls):
    cnf_loss = 0
    for b, w, h in torch.nonzero(target_cls.data):
        correct_index = target_cls.data[b, w, h] # todo
        prob = F.softmax(output_cls[b, :, w, h])[correct_index]
        p_box = output_loc[b, :, w, h]
        g_box = target_loc[b, :, w, h]
        cnf_loss += (output_cnf[b, 0, w, h] - prob * iou(p_box, g_box)) ** 2
    return cnf_loss


def yololike_loss(output, target, alpha=1, beta=1):
    output_loc, output_cnf, output_cls = output

    target_loc = target[:, :4, :, :]
    target_cls = target[:, 4, :, :].long()
    mask = torch.autograd.Variable(target_cls.data.gt(0))
    if cuda:
        mask = mask.cuda()

    loc_loss = loc_loss_func(output_loc, target_loc, mask)
    cnf_loss = cnf_loss_func(output_loc, output_cnf, output_cls, target_loc, target_cls)
    cls_loss = cls_loss_func(output_cls, target_cls)

    total = loc_loss + (alpha * cls_loss) + (beta * cnf_loss)

    return total, loc_loss, cnf_loss, cls_loss


def count_correct(output, target):

    correct = 0
    for b, w, h in torch.nonzero(target.data):
        pred_cls = output.data[b, :, w, h].max()[1]
        correct += pred_cls.eq(target.data).sum()
    return correct

if __name__ == '__main__':
    import doctest
    doctest.testmod()
