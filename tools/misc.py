# Miscenallious tools for Nanosam 2.
#


class ModelSource:
    def __init__(self, name:str, checkpoint:str, cfg:str):
        self.name = name
        self.checkpoint = checkpoint
        self.cfg = cfg