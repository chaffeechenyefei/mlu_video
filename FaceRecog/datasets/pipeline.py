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


class RandomCrop(object):
    def __init__(self, crop_ratio=[0.9, 1.0], crop_method='central', execute_ratio=1.0):
        self.crop_ratio = crop_ratio
        self.crop_method = crop_method
        self.execute_ratio = execute_ratio
        if crop_ratio is not None:
            assert min(crop_ratio) > 0 and max(crop_ratio) <= 1
        assert crop_method in ['central', 'random']

    def __call__(self, img_cv2):
        if np.random.uniform(low=0, high=1.0) <= self.execute_ratio:
            h, w = img_cv2.shape[0:2]
            ratio = np.random.uniform(low=min(self.crop_ratio), high=max(self.crop_ratio))
            new_h, new_w = int(h * ratio), int(w * ratio)
            x0 = np.random.randint(low=0, high=w - new_w)
            y0 = np.random.randint(low=0, high=h - new_h)
            img_cv2 = img_cv2[y0: y0 + new_h, x0:x0 + new_w]
            img_cv2 = cv2.resize(img_cv2, (w, h))
            return img_cv2
        else:
            return img_cv2

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={})'.format(
            self.crop_ratio)


class RandomEffect(object):
    def __init__(self):
        super(RandomEffect, self).__init__()

    def _change_contrast(self, origin_img, base=0.8, gamma=1.3):
        blank = np.zeros(origin_img.shape, origin_img.dtype)
        output_img = cv2.addWeighted(origin_img, base, blank, 1 - base, gamma)
        return output_img

    def _decrease_brightness(self, origin_img):
        weight = [0.3, 0.5, 0.6, 0.7, 0.8],
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

    def __call__(self, img_cv2, execute_ratio=0.1):
        if np.random.uniform(low=0, high=1.0) <= execute_ratio:
            img_cv2 = self._change_contrast(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= execute_ratio:
            img_cv2 = self._add_warm_light(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= execute_ratio:
            img_cv2 = self._add_random_tone(img_cv2)
        if np.random.uniform(low=0, high=1.0) <= execute_ratio:
            img_cv2 = self._decrease_brightness(img_cv2)
        return img_cv2


class AddNoise(object):
    def __init__(self):
        super(AddNoise, self).__init__()

    def _addSaltAndPepperNoise(self, src, percentage):
        SP_NoiseImg = src
        SP_NoiseNum = int(percentage * src.shape[0] * src.shape[1])
        for i in range(SP_NoiseNum):
            randX = random.randint(0, src.shape[0] - 1)
            randY = random.randint(0, src.shape[1] - 1)
            if random.randint(0, 1) == 0:
                SP_NoiseImg[randX, randY] = 0
            else:
                SP_NoiseImg[randX, randY] = 255
        return SP_NoiseImg

    def _addGaussianNoise(self, image, percentage):
        G_Noiseimg = image
        G_NoiseNum = int(percentage * image.shape[0] * image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, image.shape[0] - 1)
            temp_y = np.random.randint(0, image.shape[1] - 1)
            G_Noiseimg[temp_x][temp_y] = 255
        return G_Noiseimg

    def __call__(self, origin_img, execute_ratio=1.0):
        if np.random.uniform(low=0, high=1.0) <= execute_ratio:
            if random.randint(0, 1):
                return self._addSaltAndPepperNoise(origin_img, percentage=random.uniform(0, 0.05))
            else:
                return self._addGaussianNoise(origin_img, percentage=random.uniform(0.01, 0.1))
        else:
            return origin_img


class MixUp(object):
    def __init__(self, mix_img_dir='/data/data/', mix_ratio=[0.1, 0.3]):
        super(MixUp, self).__init__()
        self.mix_img_fpath = [
            os.path.join(mix_img_dir, img_fn)
            for img_fn in os.listdir(mix_img_dir)
            if img_fn.endswith(('.jpg', 'jpeg', 'png'))
               and os.path.isfile(os.path.join(mix_img_dir, img_fn))
        ]
        self.mix_ratio = mix_ratio

    def _crop_small_bg(self):
        for img_fpath in self.mix_img_fpath:
            print(img_fpath)




    def __call__(self, img_cv2):
        mix_img = random.choice(self.mix_img_fpath)
        mix_img = cv2.resize(mix_img, (img_cv2.shape[1], img_cv2.shape[0]))
        weight = random.uniform(a=min(self.mix_ratio), b=max(self.mix_ratio))
        img_cv2 = cv2.addWeighted(img_cv2, 1 - weight, mix_img, weight, 1.0)
        return img_cv2


class Pipeline(object):
    def __init__(self):
        super(Pipeline, self).__init__()
        self.random_flip = RandomFlip(flip_ratio=0.5, direction='horizontal')
        self.random_crop = RandomCrop(crop_ratio=[0.85, 0.95], execute_ratio=0.8)
        self.add_noise = AddNoise()
        self.random_effect = RandomEffect()

    def __call__(self, img_cv2):
        img_cv2 = self.random_flip(img_cv2)
        img_cv2 = self.random_crop(img_cv2)
        # img_cv2 = self.add_noise(img_cv2)
        # img_cv2 = self.random_effect(img_cv2)
        return img_cv2

