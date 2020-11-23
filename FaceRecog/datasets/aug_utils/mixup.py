import cv2
import numpy as np
import random
import os


class MixUp(object):
    def __init__(self,
                 mix_img_dir='/data/FaceRecog/tupu_data/mixup_background/',
                 mix_ratio=[0.1, 0.3]):
        super(MixUp, self).__init__()
        self.mix_img_fpath = [
            os.path.join(mix_img_dir, img_fn)
            for img_fn in os.listdir(mix_img_dir)
            if img_fn.endswith(('.jpg', 'jpeg', 'png'))
               and os.path.isfile(os.path.join(mix_img_dir, img_fn))
        ]
        self.mix_ratio = mix_ratio

    def _random_crop_mix_bg(self):
        img_fpath = random.choice(self.mix_img_fpath)
        img_cv2 = cv2.imread(img_fpath)
        h, w = img_cv2.shape[0:2]
        resize_ratio = np.random.uniform(low=0.2, high=0.6)
        new_w, new_h = int(w*resize_ratio), int(h*resize_ratio)
        img_cv2 = cv2.resize(img_cv2, (new_w, new_h))
        crop_size = np.random.randint(low=int(0.2*min(new_h, new_w)), high=int(0.6*max(new_h, new_w)))
        x0 = np.random.randint(low=0, high=(new_w - crop_size))
        y0 = np.random.randint(low=0, high=(new_h - crop_size))
        crop_img = img_cv2[y0:y0+crop_size, x0:x0+crop_size]
        # print('crop_size: {}'.format(crop_img.shape))
        return crop_img


    def __call__(self, img_cv2):
        crop_img = self._random_crop_mix_bg()
        crop_img = cv2.resize(crop_img, (img_cv2.shape[1], img_cv2.shape[0]))
        weight = random.uniform(a=min(self.mix_ratio), b=max(self.mix_ratio))
        img_cv2 = cv2.addWeighted(img_cv2, 1 - weight, crop_img, weight, 1.0)
        return img_cv2



if __name__ == "__main__":
    mixup = MixUp(mix_img_dir='/data/FaceRecog/tupu_data/mixup_background/')
    img_dir = '/data/FaceRecog/tupu_data/image_cache_0602_mtcnn_crop'
    for img_fn in os.listdir(img_dir):
        img_fpath = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = mixup(img_cv2)
        cv2.imwrite('/data/FaceRecog/results/' + img_fn, img_cv2)

