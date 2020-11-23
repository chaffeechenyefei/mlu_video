import cv2
import numpy as np
import math

#bgr
mu_p = [0.406, 0.485, 0.456]
std_p = [0.225, 0.229, 0.224]

def _gammaMuSolver(X,y,gamma=1.0,lr=1,max_iter = 10):
    #1/2 ||Sum_X^gamma/D - y||_F
    #delta = (Sum_X^gamma - y)*Sum{X_r*ln(X)}
    X = X.reshape(-1,1)+1e-3
    D = X.shape[0]
    lnX = np.log(X)
    for iter in range(max_iter):
        Xg = np.power(X,gamma)
        sumX = Xg.mean()-y
        sumP2 = (Xg*lnX).sum()/D
        gamma = gamma - lr*sumX*sumP2
    return gamma

class lightNormalizor(object):
    def __init__(self,do_scale = True):
        self.mu_p = mu_p
        self.std_p = std_p
        self.do_scale = do_scale
        self.gamma = []

    def run(self,img_cv2):
        if self.do_scale:
            img = cv2.resize(img_cv2,(200,150))
        else:
            img = img_cv2.copy()

        img = img.astype(np.float32)/255
        img_cv2 = img_cv2.astype(np.float32)/255
        self.gamma = self._get_gamma(img)

        self.gamma = list(map( lambda x: max(min(x,1.2),0.35) ,self.gamma))

        nimg = self._apply_gamma(img_cv2)
        nimg = (nimg*255).astype(np.uint8)
        return nimg

    def apply_gamma(self,img_cv2):
        img_cv2 = img_cv2.astype(np.float32) / 255
        nimg = self._apply_gamma(img_cv2)
        nimg = (nimg * 255).astype(np.uint8)
        return nimg


    def _get_gamma(self,img_cv2):
        nch = 3
        gSolver = lambda x: _gammaMuSolver(img_cv2[:, :, x], mu_p[x])
        return [gSolver(c) for c in range(nch)]

    def _apply_gamma(self,img_cv2):
        nch = 3
        gApplier = lambda x: np.power(img_cv2[:, :, x], self.gamma[x])
        img_g = [gApplier(c) for c in range(nch)]
        img_g = np.stack(img_g, axis=2)
        return img_g

