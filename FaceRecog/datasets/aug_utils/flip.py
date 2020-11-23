import cv2
import numpy as np
import random
import os


class RandomFlip(object):
    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, img_cv2):
        flip = True if np.random.rand() < self.flip_ratio else False
        if flip:
            if self.direction == 'horizontal':
                img_cv2 = np.flip(img_cv2, axis=1)
            else:
                img_cv2 = np.flip(img_cv2, axis=0)
        return img_cv2

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)

