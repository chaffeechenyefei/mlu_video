import cv2
import numpy as np
import os
import random


class RandomEffect2(object):

    def __init__(self):
        super(RandomEffect2, self).__init__()

    def _erode_process(self, origin_img, ksize=(3, 3)):
        dist = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        output_img = cv2.erode(origin_img, dist)
        return output_img

    def _dilate_process(self, origin_img, ksize=(3, 3)):
        dist = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        dilation = cv2.dilate(origin_img, dist)
        return dilation

    def _blur_process(self, origin_img, ksize=(3, 3)):
        blur_img = cv2.blur(origin_img, ksize)
        return blur_img

    def _mosaic_process(self, origin_img, mosaic_weight=0.5):
        assert mosaic_weight > 0.1 and mosaic_weight < 1.0
        h, w = origin_img.shape[0:2]
        new_h, new_w = int((1-mosaic_weight) * h), int((1-mosaic_weight) * w)
        mosaic_img = cv2.resize(origin_img, (new_w, new_h))
        return cv2.resize(mosaic_img, (w, h))

    def __call__(self, img_cv2):
        if np.random.rand() < 0.1:
            img_cv2 = self._erode_process(img_cv2)
        if np.random.rand() < 0.1:
            img_cv2 = self._dilate_process(img_cv2)
        if np.random.rand() < 0.2:
            if np.random.rand() < 0.5:
                img_cv2 = self._blur_process(img_cv2)
            else:
                img_cv2 = self._mosaic_process(
                    img_cv2,
                    mosaic_weight=np.random.uniform(low=0.1, high=0.3)
                )
        return img_cv2


if __name__ == "__main__":
    func = RandomEffect2()
    img_dir = '/data/FaceRecog/tupu_data/image_cache_0602_mtcnn_crop'
    for img_fn in os.listdir(img_dir):
        img_fpath = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = func(img_cv2)
        cv2.imwrite('/data/FaceRecog/results/aug_test/' + img_fn, img_cv2)


