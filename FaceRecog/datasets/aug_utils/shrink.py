import cv2
import numpy as np


class RandomCrop(object):
    def __init__(self, crop_ratio=[0.85, 0.95], crop_method='central'):
        self.crop_ratio = crop_ratio
        self.crop_method = crop_method
        if crop_ratio is not None:
            assert min(crop_ratio) > 0 and max(crop_ratio) <= 1
        assert crop_method in ['central', 'random']

    def __call__(self, img_cv2):
        h, w = img_cv2.shape[0:2]
        ratio = np.random.uniform(low=min(self.crop_ratio), high=max(self.crop_ratio))
        new_h, new_w = int(h * ratio), int(w * ratio)
        x0 = np.random.randint(low=0, high=w - new_w)
        y0 = np.random.randint(low=0, high=h - new_h)
        img_cv2 = img_cv2[y0: y0 + new_h, x0:x0 + new_w]
        img_cv2 = cv2.resize(img_cv2, (w, h))
        return img_cv2


    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={})'.format(
            self.crop_ratio)