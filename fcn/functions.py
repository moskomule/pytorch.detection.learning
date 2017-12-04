import torch
from torchvision import transforms

import matplotlib.pyplot as plt

from PIL import Image
import imagenet_tool


def load_image(image_path, image_size=None):
    image = Image.open(image_path)
    if image_size is not None:
        image = image.resize(image_size)
    return transforms.ToTensor()(image)


def build_heatmap(output, synset):
    # output [1, c, w, h] -> [c, w, h]
    output = output.squeeze(0).numpy()
    ids = imagenet_tool.synset_to_dfs_ids(synset)
    ids = [id_ for id_ in ids if id_ is not None]
    # sums up along c of ids
    output = output[ids, :, :].sum(axis=0)
    return torch.FloatTensor(output).unsqueeze(0)


def show_heatmap(heat_map, original_path, mode="save"):
    original = Image.open(original_path)
    heat_map = heat_map.numpy()[0]

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(original)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(heat_map)
    plt.axis("off")
    if mode == "show":
        plt.show()
    else:
        plt.savefig("{}-fcn.jpg".format(original_path.split(".")[0]))
