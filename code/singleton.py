import numpy as np


class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.kernel = np.array([[0, 0, 1, 1, 0, 0],
                                [0, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 1, 0],
                                [0, 0, 1, 1, 0, 0]], dtype=np.uint8)
        self.rad_thresh = 15
        self.roi2 = None
        self.roi2_init = None
