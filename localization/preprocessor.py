import os
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from data_storage import class_id

from load_xml import xmls_to_list


def which_grid(position, grid_size, scales):
    """
    returns grid's loc the given point should be in
    """
    x, y = position
    _x = x * scales[0] // grid_size
    _y = y * scales[1] // grid_size
    return int(_x), int(_y)


def normalize_pos(position, original_size):
    """
    normalize positions in (0, 1)
    """
    x, y, w, h = position
    im_w, im_h = original_size
    # max(w) = im_w/2
    return x / im_w, y / im_h, 2 * w / im_w, 2 * h / im_h


def generate_data(annotation: dict, image_size: int=150,
                  grid_num: int=5, image_dir=""):
    """
    generate data from annotation dict 
    annotation 
    {"filename": image_name,
    "size": (im_width, im_height),
    "objects": obj_list}
    
    :returns image_tensor(3, image_size, image_size), 
             target_tensor(**, grid, grid),
             original image size,
             scales
    """

    image_name = os.path.join(image_dir, annotation["filename"])
    grid_size = image_size // grid_num
    scales = (image_size / annotation["size"][0],
              image_size / annotation["size"][1])

    to_torch = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_name).resize((image_size, image_size))
    image_tensor = to_torch(image)

    # torch[:,x,x] = [1(bbox+conf.)+20class]
    target_tensor = torch.zeros([625, grid_num, grid_num])
    for i, (name, pos) in enumerate(annotation["objects"]):
        obj_class = class_id[name]
        grid_x, grid_y = which_grid(pos[0:2], grid_size, scales)
        x, y, w, h = normalize_pos(pos, annotation["size"])
        target_tensor[5+obj_class, grid_x, grid_y] = 1
        target_tensor[0: 5, grid_x, grid_y] = torch.FloatTensor([x, y, w, h, 1])

    return image_tensor, target_tensor, annotation["size"], scales


def get_annotaions(dir, test_size=0.1):
    alist = xmls_to_list(dir)
    train, test = train_test_split(alist, test_size=test_size)
    return train, test


class VocDataSet:

    def __init__(self, annotations, image_size=150, grid_num=5, image_dir=""):
        """ 
        DataSet of VOC for DataLoader
        """
        self.annotations = annotations
        self.image_size = image_size
        self.grid_num = grid_num
        self.image_dir = image_dir

    def __getitem__(self, index):
        return generate_data(self.annotations[index], self.image_size,
                             self.grid_num, self.image_dir)

    def __len__(self):
        return len(self.annotations)
