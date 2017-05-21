import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from functions import *

# use resnet101
resnet = models.resnet101(pretrained=True)
reshead = nn.Sequential(*list(resnet.children())[:-2])


class Fully_Conv_ResNet(nn.Module):
    def __init__(self):
        super(Fully_Conv_ResNet, self).__init__()
        # base model is a res50 model before GlobalAveragePooling layer
        # the output filters of base_model is 2048
        self.base_model = reshead
        self.conv = nn.Conv2d(2048, 1000, kernel_size=1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.conv(x)
        return filter_softmax(x)


def main(image_path, synset_id, mode="save", image_size=None):
    fc_net = Fully_Conv_ResNet()
    # load pre-trained weights for base
    fc_net.base_model.load_state_dict(reshead.state_dict())
    # load pre-trained fc layer's weights to new convolution's kernel
    # since kernel tensor is 4d, reshape is needed
    fc_net.conv.load_state_dict({"weight": resnet.fc.state_dict()["weight"]
                                .view(1000, 2048, 1, 1),
                                 "bias": resnet.fc.state_dict()["bias"]})

    input = load_image(image_path, image_size)
    # batch size = 1
    input = Variable(input).unsqueeze(0)
    # turn on eval mode not to use batch-normalization
    # dropout like training phase
    fc_net.eval()
    output = fc_net(input)
    hmap = build_heatmap(output.data, synset_id)
    show_heatmap(hmap, image_path, mode)

if __name__ == '__main__':
    dog_s = "n02084071"
    cat_s = "n02121808"
    main("cat.jpg", cat_s)
    main("cat2.jpg", cat_s)
    main("cat3.jpg", cat_s)
    main("cat4.jpg", cat_s)
    main("dog.jpg", dog_s)
    main("dog2.jpg", cat_s)
