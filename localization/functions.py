import torch
import torch.nn.functional as F
from numba.decorators import jit


cuda = torch.cuda.is_available()


def iou(pred, gt):
    """
    >>> a=iou([0.4, 0.5, 0.1, 0.2], [0.5, 0.5, 0.1, 0.2])
    >>> (a-1/3) < 1e-2
    True
    >>> iou([0.4, 0.5, 0.1, 0.2], [0.1, 0.5, 0.1, 0.2])
    0.0
    """
    px_0 = pred[0] - pred[2]
    px_1 = pred[0] + pred[2]
    py_0 = pred[1] - pred[3]
    py_1 = pred[1] + pred[3]

    gx_0 = gt[0] - gt[2]
    gx_1 = gt[0] + gt[2]
    gy_0 = gt[1] - gt[3]
    gy_1 = gt[1] + gt[3]

    x_0 = max(px_0, gx_0)
    x_1 = min(px_1, gx_1)
    y_0 = max(py_0, gy_0)
    y_1 = min(py_1, gy_1)

    iarea = max(x_1 - x_0, 0) * max(y_1 - y_0, 0)
    parea = 4 * pred[2] * pred[3]
    garea = 4 * gt[2] * gt[3]

    return iarea/(parea+garea-iarea)

@jit
def cl_func(output, target):
    # class loss function
    loss = 0
    for b, w, h in torch.nonzero(target.data):
        loss += F.cross_entropy(output[b, :, w, h].contiguous().view(1, -1), target[b, w, h])
    return loss


@jit
def ll_func(output_loc, output_cls, target_loc, target_cls):
    # location loss function
    loc_loss = 0
    cnf_loss = 0
    for b, w, h in torch.nonzero(target_cls.data):
        correct_index = target_cls.data[b, w, h]
        prob = F.softmax(output_cls[b, correct_index, w, h])
        p_box = output_loc[b, :4, w, h]
        g_box = target_loc[b, :, w, h]
        loc_loss += torch.sum((p_box - g_box) ** 2)
        cnf_loss += (output_loc[b, 4, w, h] - prob * iou(p_box, g_box)) ** 2

    return loc_loss, cnf_loss


@jit
def yololike_loss(output, target, alpha=10, beta=1):
    output_loc, output_cls = output

    target_loc = target[:, :4, :, :]
    target_cls = target[:, 4, :, :].long() # just class label

    l_loss, cnf_loss = ll_func(output_loc, output_cls, target_loc, target_cls)
    c_loss = cl_func(output_cls, target_cls)
    total = l_loss + (alpha * c_loss) + (beta * cnf_loss)

    return total, l_loss, c_loss


def count_correct(output, target):

    correct = 0
    for b, w, h in torch.nonzero(target.data):
        pred_cls = output.data[b, :, w, h].max()[1]
        correct += pred_cls.eq(target.data).sum()
    return correct

if __name__ == '__main__':
    import doctest
    doctest.testmod()
