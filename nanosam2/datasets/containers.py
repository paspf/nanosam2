# Containers for different use cases for Nanosam2.
#


class ModelSource:
    def __init__(self, name:str, checkpoint:str, cfg:str, img_shape:list[int]):
        self.name = name
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.img_shape = img_shape
