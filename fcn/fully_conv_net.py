from torch import nn
from torchvision import models


class FCResNet(nn.Module):
    def __init__(self):
        super(FCResNet, self).__init__()
        # base model is a res50 model before GlobalAveragePooling layer
        # the output filters of base_model is 2048
        self._resnet = models.resnet101(pretrained=True)
        self.base_model = nn.Sequential(*list(self._resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, 1000, kernel_size=1)
        self.softmax2d = nn.Softmax2d()

    def forward(self, x):
        x = self.base_model(x)
        x = self.conv(x)
        return self.softmax2d(x)

    def load_weight(self):
        original_fc = self._resnet.fc.state_dict()
        # load pre-trained fc layer's weights to new convolution's kernel
        # since kernel tensor is 4d, reshape is needed
        self.conv.load_state_dict({"weight": original_fc["weight"].view(1000, 2048, 1, 1),
                                   "bias": original_fc["bias"]})
