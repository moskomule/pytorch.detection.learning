import torch

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


def yololike_loss(data, target):
    pass


def filter_softmax(x):
    """
    softmax along filters
    """

    assert x.dim() == 4, "dimension of input must be 4"
    _, f, _, _ = x.size()
    nom = torch.exp(x)
    dinom = torch.sum(nom, 1).repeat(1, f, 1, 1)
    return nom/dinom

if __name__ == '__main__':
    import doctest
    doctest.testmod()