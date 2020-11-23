import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from retina_infer import RetinaFaceDet
from functools import reduce
from dataset.plateloader import BasicLoader
from torch.utils.data import DataLoader

pj = os.path.join

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.7, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    cudnn.benchmark = True
    # args.cpu = True
    faceDet = RetinaFaceDet(args.network,args.trained_model,use_cpu=args.cpu)
    faceDet.set_default_size([1920,1080,3])

    image_path = "/data/yefei/data/0107_0114/"
    save_path = "/data/yefei/data/0107_0114_result/"
    # image_path = "./data/shuyan/"
    # save_path = "./data/shuyan/result/"

    imgext = '.jpg'

    batch_size = 8

    dataset_test = BasicLoader(imgs_dir=image_path,extstr='.jpg')
    dataloader_test = DataLoader(dataset=dataset_test, num_workers = 4, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for cnt,batch in enumerate(dataloader_test):
            img_data = batch['image']
            img_names = batch['name']

            detss = faceDet.execute_batch(img_data, threshold=0.6, topk=args.top_k, keep_topk=args.keep_top_k,
                                          nms_threshold=args.nms_threshold)

            for bnt, dets in enumerate(detss):
                img_raw = cv2.imread(pj(image_path, img_names[bnt]))
                for idx, b in enumerate(dets):
                    if b[4] < args.vis_thres:
                        continue
                    b = list(map(int, b))
                    scale = 1.3
                    pwidth = int((b[2] - b[0]) * scale)
                    pheight = int((b[3] - b[1]) * scale)
                    pcx = int((b[2] + b[0]) / 2)
                    pcy = int((b[3] + b[1]) / 2)
                    img_face = cv2.getRectSubPix(img_raw, (pwidth, pheight), (pcx, pcy))
                    img_face = format(img_face, 112)
                    # img_face = cv2.resize(img_face,(112,112))

                    savename = [c for c in img_names[bnt].split(imgext) if c][0]
                    savename = '%s_#%d.jpg' % (savename, idx)
                    savefullname = pj(save_path, savename)

                    cv2.imwrite(savefullname, img_face)





    # imglst = os.listdir(image_path)
    # imglst = [ c for c in imglst if c.endswith(imgext)]
    # print('Total #%d imgs...'%len(imglst))
    # for cnt,imgname in enumerate(imglst):
    #     if cnt%100==0:
    #         print('%1.3f done...'%(cnt/len(imglst)))
    #
    #     imgfullname = pj(image_path,imgname)
    #     img_raw = cv2.imread(imgfullname, cv2.IMREAD_COLOR)
    #     img_raw = cv2.resize(img_raw,(1920,1080))
    #     if img_raw is not None:
    #         dets = faceDet.execute(img_raw, threshold=0.6, topk=args.top_k, keep_topk=args.keep_top_k,
    #                                nms_threshold=args.nms_threshold)
    #         for idx,b in enumerate(dets):
    #             if b[4] < args.vis_thres:
    #                 continue
    #             b = list(map(int, b))
    #             # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #             scale = 1.3
    #             pwidth = int((b[2] - b[0])*scale)
    #             pheight = int((b[3] - b[1])*scale)
    #             pcx = int((b[2] + b[0])/2)
    #             pcy = int((b[3] + b[1])/2)
    #             img_face = cv2.getRectSubPix(img_raw,(pwidth,pheight),(pcx,pcy))
    #             img_face = format(img_face,112)
    #             # img_face = cv2.resize(img_face,(112,112))
    #
    #             savename = [ c for c in imgname.split(imgext) if c][0]
    #             savename = '%s_#%d.jpg'%(savename,idx)
    #             savefullname = pj(save_path,savename)
    #
    #             cv2.imwrite(savefullname, img_face)
