import cv2
import numpy as np
import random
import os


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

    def __call__(self, origin_img):
        if random.randint(0, 1):
            return self._addSaltAndPepperNoise(origin_img, percentage=random.uniform(0, 0.05))
        else:
            return self._addGaussianNoise(origin_img, percentage=random.uniform(0.01, 0.1))


