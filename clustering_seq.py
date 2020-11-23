"""
<<!args.label>>
Do clustering sequentially which should be alike the real situation
input:
    {timestamp}.pkl
        [db_feat,db_imgpath,db_attr]
        db_feat = np.array((B,512))
        db_imgpath = list[full path of image]
        db_attr = list[{cam_name}] e.g cam_name = '10B_6_3'
output:
    {
        cluster_id: {'cluster':[ list of image names ] , 'pred': str, name of prediction , 'score': float, similarity }
    }
<<args.label>>
output:
    images for labelling.
"""
import numpy as np
import sys,os,pickle,cv2,argparse,shutil
from sklearn.preprocessing import normalize
from udftools.functions import largest_indices
from sklearn.cluster import AgglomerativeClustering
from udftools.functions import unsupervised_cluster

pj = os.path.join

def save_obj(obj, name ):
    with open( name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--cam',default='10B_6_0')
    parser.add_argument('--gallery',default=None,help='if gallery is given, then each folder will return top5 most similar person inside the gallery')
    parser.add_argument('--topsim', type=float, default=0.30)
    parser.add_argument('--top',type=int,default=1)
    parser.add_argument('--K',type=int,default=10)
    parser.add_argument('--save')
    parser.add_argument('--label',action='store_true',help='')
    parser.add_argument('--all',action='store_true',help='')
    parser.add_argument('--method',default='sklearn',help='using sklearn or self')

    args = parser.parse_args()

    print('*'*15)
    print('*' * 5, os.path.basename(args.data))
    print('*'*5, args.cam )
    print('*' * 15)

    if args.method == 'sklearn':
        print('using AgglomerativeClustering...')
    else:
        print('using self unsupervised clustering...')

    #slice_rec :: !args.label
    """
    {
        cluster_id: {'cluster':[ list of image names ] , 'pred': str, name of prediction , 'score': float, similarity }
    }
    """
    slice_rec = {}

    #save path
    timestamp_folder = (os.path.basename(args.data)).replace('.pkl','')

    if args.label:
        savepath = pj(args.save,timestamp_folder)
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        savepath = pj(savepath,args.cam)
        if not os.path.exists(savepath):
            os.mkdir(savepath)
    else:
        savepath = args.save
        savename = '%s@%s@slice_prediction.pkl'%(timestamp_folder,args.cam)

    FLG_instance = False
    if args.K == 1:
        FLG_instance = True


    gfeat = []
    gpath = []
    # loading gallery
    if os.path.exists(args.gallery):
        db_gallery = load_obj(args.gallery)

        for person_name in db_gallery.keys():
            gfeat.append(db_gallery[person_name]['feat'])
            gpath.append(db_gallery[person_name]['imgpath'])

        glabel = list(db_gallery.keys())
        glabel = np.array(glabel)

        gfeat = np.concatenate(gfeat, axis=0)
        gfeat = normalize(gfeat, axis=1)
        print('gallery loaded: shape of (gfeat,glabel):')
        print(gfeat.shape, glabel.shape)
    # end loading gallery
    ##

    #rerank the sequence according to frame
    db_full = pj(args.data)
    [db_feat, db_imgpath, db_attr] = load_obj(db_full)
    db_attr = np.array(db_attr)
    db_imgpath = np.array(db_imgpath)
    target_id = db_attr == args.cam

    sub_db_feat = db_feat[target_id,:]
    sub_db_imgpath = db_imgpath[target_id]
    print('sub data size = %d'%len(sub_db_feat))
    sub_db_feat = normalize(sub_db_feat,axis=1)

    db_frame = []
    for c in sub_db_imgpath:
        imgname = os.path.basename(str(c))
        #frame#67317#2020-09-02#13:44:53#0#.jpg
        frame_num = int(imgname.split('#')[1])
        db_frame.append(frame_num)

    db_frame = np.array(db_frame)

    rankid = db_frame.argsort()

    sub_db_feat = sub_db_feat[rankid,:]
    sub_db_imgpath = sub_db_imgpath[rankid]
    db_frame = db_frame[rankid]

    start_pos = 0
    cluster_cnt = 0
    slice_cnt = 0

    FLG_do = True
    while FLG_do:
        print('%d'%start_pos, end='\r')
        if args.all:
            bfeat = sub_db_feat
            bframe = db_frame
            bimgpath = sub_db_imgpath
            FLG_do = False
        else:
            bfeat = sub_db_feat[start_pos:start_pos+args.K]
            bframe = db_frame[start_pos:start_pos + args.K]
            bimgpath = sub_db_imgpath[start_pos:start_pos + args.K]

        start_pos += len(bfeat)

        if bfeat.shape[0] < 2:
            FLG_do = False
            break

        slice_cnt += 1

        if args.all:
            if args.method == 'sklearn':
                print('AgglomerativeClustering...')
            else:
                print('self unsupervised clustering...')

        if args.method == 'sklearn':
            model = AgglomerativeClustering(linkage='average', distance_threshold=1.0, n_clusters=None)
        else:
            model = unsupervised_cluster( distance_threshold=0.5)

        model = model.fit(bfeat)

        label = model.labels_
        n_cluster = model.n_clusters_

        if args.all:
            print('Assigning each cluster a label...')

        FLG_save_cluster = True
        for i in range(n_cluster):
            c_label_id = label == i
            i_cls_feat = bfeat[c_label_id]
            i_cls_feat = i_cls_feat.mean(axis=0)
            i_cls_imgpath = bimgpath[c_label_id]

            pgMtx = i_cls_feat @ gfeat.transpose()
            pgMtx = pgMtx.reshape(-1)
            topid = largest_indices(pgMtx, args.top)[0]


            if args.label:
                if pgMtx[topid[0]] > args.topsim:
                    FLG_save_cluster = True
                else:
                    FLG_save_cluster = False

                if FLG_save_cluster:
                    # print('%d-th cluster' % cluster_cnt, end='\r')
                    save_cluster_path_full = pj(savepath, '%05d'%cluster_cnt)
                    if not os.path.exists(save_cluster_path_full):
                        os.mkdir(save_cluster_path_full)
                    for id in topid:
                        img_name = os.path.basename(gpath[id])
                        img_name = 'gallery_%0.3f_%s' % (pgMtx[id], img_name)
                        # shutil.copyfile(gpath[id], pj(save_cluster_path_full, img_name))
                        cv2_img = cv2.imread(gpath[id])
                        cv2_img = cv2.resize(cv2_img,dsize=(224,224))
                        cv2.imwrite(pj(save_cluster_path_full, img_name),cv2_img)

                    for v in i_cls_imgpath:
                        img_name = os.path.basename(v)
                        shutil.copyfile(v, pj(save_cluster_path_full, img_name))

                    cluster_cnt+=1
            else:
                top1_id = topid[0]
                gallery_img_name = os.path.basename(gpath[top1_id])
                gallery_img_name = gallery_img_name.split('_')[-1].replace('.jpg','') #{Dep}-{Name}
                slice_img_name = [ os.path.basename(c) for c in i_cls_imgpath ]
                score = pgMtx[top1_id]

                slice_rec[cluster_cnt] = {
                    'cluster':slice_img_name,
                    'pred':gallery_img_name,
                    'score':score,
                }

                cluster_cnt += 1


    print('[No use]Statistic:%d/%d = %1.3f'%(cluster_cnt,slice_cnt,cluster_cnt / slice_cnt))

    if not args.label:
        save_obj(slice_rec, pj(savepath,savename) )





