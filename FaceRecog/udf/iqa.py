import cv2
import numpy as np
import os
import pickle
import math

pj = os.path.join


def crop_margin(img_cv, margin=0.2):
    h, w = img_cv.shape[0:2]
    margin = math.floor(margin * min(h, w))

    crop_img_cv = img_cv[margin:h - margin, margin:w - margin].copy()
    return crop_img_cv


def crop_margin_col(img_cv, margin=0.2):
    h, w = img_cv.shape[0:2]
    margin = math.floor(margin * min(h, w))

    crop_img_cv = img_cv[:, margin:w - margin].copy()
    return crop_img_cv


def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    """
    im2col_sliding_broadcasting
    :param A: [H,W]
    :param BSZ: [winH,winW]
    :param stepsize: 
    :return: [winH,winW,N]
    """
    # Parameters
    M, N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:, None] * N + np.arange(BSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel()[::stepsize]).reshape(BSZ[1], BSZ[0], -1)

def im2col(mtx, block_size, stepsize=1):
    '''
    block_size = W(H)
    '''
    mtx_shape = mtx.shape
    sx = math.floor((mtx_shape[0] - block_size + 1) / stepsize)
    sy = math.floor((mtx_shape[1] - block_size + 1) / stepsize)
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size, block_size, sx * sy))
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:, :, i * sx + j] = mtx[j * stepsize:j * stepsize + block_size,
                                       i * stepsize:i * stepsize + block_size]
    return result


def col2im(mtx, image_size, block_size, stepsize=1):
    sx = math.floor((image_size[0] - block_size + 1) / stepsize)
    sy = math.floor((image_size[1] - block_size + 1) / stepsize)
    result = np.zeros(image_size)
    weight = np.zeros(image_size) + 1e-3  # weight记录每个单元格的数字重复加了多少遍
    col = 0
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[j * stepsize:j * stepsize + block_size, i * stepsize:i * stepsize + block_size] += mtx[:, :, col]
            weight[j * stepsize:j * stepsize + block_size, i * stepsize:i * stepsize + block_size] += np.ones(
                block_size)
            col += 1
    return result / weight


def col2im_mtx_single_value(mtx, image_size, block_size, stepsize=1):
    sx = math.floor((image_size[0] - block_size + 1) / stepsize)
    sy = math.floor((image_size[1] - block_size + 1) / stepsize)

    nimage_size = [(sx - 1) * stepsize + block_size, (sy - 1) * stepsize + block_size]
    result = np.zeros(nimage_size)
    weight = np.zeros(nimage_size) + 1e-3  # weight记录每个单元格的数字重复加了多少遍
    col = 0
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[j * stepsize:j * stepsize + block_size, i * stepsize:i * stepsize + block_size] += mtx[col]
            weight[j * stepsize:j * stepsize + block_size, i * stepsize:i * stepsize + block_size] += np.ones(
                block_size)
            col += 1
    return result / weight


def compute_smap(img, blocksize, stepsize):
    patches = im2col(img, blocksize, stepsize)
    imgsize = img.shape
    n_patches = patches.shape[-1]
    mtx = []
    for n in range(n_patches):
        p = patches[:, :, n]
        s = np.linalg.svd(p, full_matrices=1, compute_uv=0)
        s = s[0] / s.sum()
        mtx.append(s)
    mtx = np.array(mtx)
    smap = col2im_mtx_single_value(mtx, imgsize, blocksize, stepsize)
    return smap


def get_texture_score(img, blocksize, stepsize):
    smap = compute_smap(img, blocksize, stepsize)
    return smap.mean()
