voc_class = ["nothing", "person", "bird", "cat", "cow", "dog", "horse", "sheep",
             "aeroplane", "bicycle", "boat", "bus", "car", "motorbike",
             "train", "bottle", "chair", "diningtable", "pottedplant",
             "sofa", "tvmonitor"]
class_id = {k: v for v, k in enumerate(voc_class)}
id_class = {k: v for k, v in enumerate(voc_class)}