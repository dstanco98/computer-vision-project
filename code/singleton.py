# In this file I set some constant value for the project, such as the kernel filter for the ball, the radius threshold
# the Region of Interest (ROI) and the previous one pass (?).
# NOTE: I decided to use a Singleton since it controls the object creation by
# ensuring that only one instance of the class is created.

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
        self.vel = 0
        self.prev_vel = 0
