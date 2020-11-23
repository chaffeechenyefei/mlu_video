import cv2
import numpy as np
import random
import os


class RandomEffect(object):
    def __init__(self):
        super(RandomEffect, self).__init__()

    def _change_contrast(self, origin_img, base=0.8, gamma=1.3):
        blank = np.zeros(origin_img.shape, origin_img.dtype)
        output_img = cv2.addWeighted(origin_img, base, blank, 1 - base, gamma)
        return output_img

    def _decrease_brightness(self, origin_img):
        weight = [0.3, 0.5, 0.6, 0.7, 0.8]
        background = [10, 30]
        blank = np.ones(origin_img.shape, origin_img.dtype) * random.randint(a=min(background), b=max(background))
        weight = random.choice(weight)
        output_img = cv2.addWeighted(origin_img, 1 - weight, blank, weight, 1.0)
        return output_img

    def _add_warm_light(self, origin_img, intensity=0.1):
        # origin_img: GBR
        blank = np.zeros(origin_img.shape, origin_img.dtype)
        blank[:, :, :] = [100, 238, 247]
        output_img = cv2.addWeighted(origin_img, 1 - intensity, blank, intensity, 1.0)
        return output_img

    def _add_random_tone(self, origin_img, ratio=0.1):
        # origin_img: GBR
        warm_light = np.zeros(origin_img.shape, origin_img.dtype)
        warm_light[:, :, :] = [100, 238, 247]
        dark_blue = np.zeros(origin_img.shape, origin_img.dtype)
        dark_blue[:, :, :] = [102, 51, 0]
        orange = np.zeros(origin_img.shape, origin_img.dtype)
        orange[:, :, :] = [0, 102, 204]
        light_coffe = np.zeros(origin_img.shape, origin_img.dtype)
        light_coffe[:, :, :] = [102, 153, 204]
        dark_pink = np.zeros(origin_img.shape, origin_img.dtype)
        dark_pink[:, :, :] = [204, 153, 204]
        tone = random.choice([warm_light, dark_blue, orange, light_coffe, dark_pink])
        output_img = cv2.addWeighted(origin_img, 1 - ratio, tone, ratio, 1.0)
        return output_img

    def __call__(self, img_cv2):
        if np.random.uniform(low=0, high=1.0) <= 0.2:
            img_cv2 = self._change_contrast(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= 0.1:
            img_cv2 = self._add_warm_light(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= 0.1:
            img_cv2 = self._add_random_tone(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= 0.1:
            img_cv2 = self._decrease_brightness(img_cv2)
        return img_cv2




if __name__ == "__main__":
    func = RandomEffect()
    img_dir = '/data/FaceRecog/tupu_data/image_cache_0602_mtcnn_crop'
    for img_fn in os.listdir(img_dir):
        img_fpath = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = func(img_cv2)
        cv2.imwrite('/data/FaceRecog/results/aug_test/' + img_fn, img_cv2)

