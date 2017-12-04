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
    x = x * scales[0] // grid_size
    y = y * scales[1] // grid_size
    return int(x), int(y)


def normalize_pos(position, scales, grids, grid_size, image_size):
    """
    normalize positions in (0, 1)
    """
    x, y, w, h = position
    grid_x, grid_y = grids
    x, w = x * scales[0], w * scales[0]
    y, h = y * scales[1], h * scales[1]
    # grid centers
    gc_x, gc_y = (grid_x+0.5) * grid_size, (grid_y+0.5) * grid_size
    x = 2 * (x-gc_x) / grid_size  # max x is grid_size/2
    y = 2 * (y-gc_y) / grid_size  # max x is grid_size/2
    w, h = 2 * w / image_size, 2 * h / image_size # max w, h are image_size/2
    return x, y, w, h


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
    scales = (image_size / annotation["size"][0],  # width
              image_size / annotation["size"][1])  # height

    to_torch = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_name).resize((image_size, image_size))
    image_tensor = to_torch(image)

    # torch[:,x,x] = [1*bbox+class_size+1]
    target_tensor = torch.zeros([24, grid_num, grid_num])
    for i, (name, pos) in enumerate(annotation["objects"]):
        grid_x, grid_y = which_grid(pos[0:2], grid_size, scales)
        x, y, w, h = normalize_pos(pos, scales, (grid_x, grid_y), grid_size, image_size)
        target_tensor[4+class_id[name], grid_x, grid_y] = 1
        target_tensor[0: 4, grid_x, grid_y] = torch.FloatTensor([x, y, w, h])

    return image_tensor, target_tensor, annotation["size"], scales


def get_annotations(dir, test_size=0.1):
    dir = os.path.join(dir, "Annotations/")
    alist = xmls_to_list(dir)
    train, test = train_test_split(alist, test_size=test_size)
    return train, test


class VocDataSet:

    def __init__(self, annotations, image_size=150, grid_num=5, dir=""):
        """ 
        DataSet of VOC for DataLoader
        """
        self.annotations = annotations
        self.image_size = image_size
        self.grid_num = grid_num
        self.image_dir = os.path.join(dir, "JPEGImages/")

    def __getitem__(self, index):
        return generate_data(self.annotations[index], self.image_size,
                             self.grid_num, self.image_dir)

    def __len__(self):
        return len(self.annotations)
