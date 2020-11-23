import os
import numpy as np
import matplotlib.pyplot as plt




def plot_TPR_FAR(TPR_vals, FAR_vals, thresh_vals, save_fpath='TPR_FAR.jpg'):

    plt.title('TPR-FAR')
    plt.xlabel('thresh')
    plt.xlim(0, 1)

    plt.ylim(0, 1)

    x = np.asarray(thresh_vals, dtype=np.float32).clip(min=0.0, max=1.0)
    y_TPR = np.asarray(TPR_vals, dtype=np.float32).clip(min=0.0, max=1.0)
    y_FAR = np.asarray(FAR_vals, dtype=np.float32).clip(min=0.0, max=1.0)

    plt.plot(x, y_TPR, ls='-', linewidth=1.0, label='TPR')
    plt.plot(x, y_FAR, ls='-', linewidth=1.0, label='FAR')
    plt.legend(['TPR', 'FAR'], loc='lower right')

    plt.savefig(save_fpath)
