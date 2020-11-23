import cv2
import numpy as np
import os

from datasets.aug_utils.flip import RandomFlip
from datasets.aug_utils.shrink import RandomCrop
from datasets.aug_utils.noise import AddNoise
from datasets.aug_utils.effect import RandomEffect
from datasets.aug_utils.effect2 import RandomEffect2


class Pipeline2(object):
    def __init__(self):
        super(Pipeline2, self).__init__()
        self.random_flip = RandomFlip(flip_ratio=0.5, direction='horizontal')
        self.random_crop = RandomCrop(crop_ratio=[0.80, 0.95])
        self.add_noise = AddNoise()
        self.random_effect = RandomEffect()
        self.random_effect2 = RandomEffect2()

    def __call__(self, img_cv2):
        img_cv2 = self.random_flip(img_cv2)
        if np.random.rand() < 0.5:
            img_cv2 = self.random_crop(img_cv2)
        if np.random.rand() < 0.1:
            img_cv2 = self.add_noise(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2 = self.random_effect2(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2 = self.random_effect(img_cv2)

        return img_cv2



if __name__ == "__main__":
    func = Pipeline2()
    img_dir = '/data/FaceRecog/tupu_data/image_cache_0602_mtcnn_crop'
    for img_fn in os.listdir(img_dir):
        img_fpath = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = func(img_cv2)
        cv2.imwrite('/data/FaceRecog/results/aug_test/' + img_fn, img_cv2)