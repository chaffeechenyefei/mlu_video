"""
extract features inside each folder and cluster inside
{time}/{cam}/{images} : 1596592801/10B_6_3/*.jpg
feature is stored in {time}.pkl
{cam}.pkl: dict:{'image_name':np.array()}

cluster:
X = [B,D] Y = [B,1(str)]
X = X.unit(dim=1)
sklearn.cluster.AgglomerativeClustering
"""
import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)

import torch,argparse,shutil
import numpy as np
import cv2
import pickle
from functools import reduce
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from udftools.functions import largest_indices


pj = os.path.join

def save_obj(obj, name ):
    with open( name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--save')
    parser.add_argument('--threshold',default=1.0,type=float)
    parser.add_argument('--linkage',default='ward',help='linkage{“ward”, “complete”, “average”, “single”}')
    parser.add_argument('--gallery',default=None,help='if gallery is given, then each folder will return top5 most similar person inside the gallery')
    parser.add_argument('--top',type=int,default=5,help='each cluster will be assigned with top X person from gallery')
    parser.add_argument('--topsim',type=float,default=0.30)
    parser.add_argument('--restart',action='store_true',help='If true, all the features will be extracted and refresh the extracted ones in savepath')

    args = parser.parse_args()

    # cluster_attr = ['10B_3','10B_4','10B_5','10B_6','10B_7']
    cluster_attr = ['10B_6', '10B_7']
    exclude_attr = ['10B_6_3']

    datapath = args.data
    savepath = args.save

    max_len = 25000

    gfeat = []
    glabel = []
    gpath = []

    ##
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


    if os.path.isdir(savepath):
        pass
    else:
        os.mkdir(savepath)

    db_times = [ c for c in os.listdir(datapath) if c.endswith('.pkl') ]

    #merge folders
    n_db_times = {}
    for db_time in db_times:
        if len(n_db_times) == 0:
            n_db_times[db_time] = [db_time]
        else:
            find_nearest = False
            for k in n_db_times.keys():
                if not find_nearest:
                    vlst = n_db_times[k]
                    for v in vlst:
                        if abs(int(db_time.replace('.pkl','')) - int(v.replace('.pkl',''))) < 3600*6:
                            n_db_times[k].append(db_time)
                            find_nearest = True
                            break
                else:
                    break

            if not find_nearest:
                n_db_times[db_time] = [db_time]

    # print(n_db_times)
    # exit(-1)

    #resuming
    past_lst = [ '%s.pkl'%c for c in os.listdir(savepath) if os.path.isdir(pj(savepath,c)) ]
    cand_lst = []
    if not args.restart:
        for c in n_db_times.keys():
            twoSetJoin = set(n_db_times[c]) & set(past_lst)
            # print(n_db_times[c],twoSetJoin)
            if len(twoSetJoin) == 0:
                cand_lst.append(c)
    else:
        cand_lst = [c for c in list(n_db_times.keys())]

    # print(cand_lst)
    # exit(-1)

    if len(cand_lst) == 0:
        print('All had been done before, program will exit.')
        exit(0)

    for cnt_db_time, db_time_key in enumerate(cand_lst):
        print('#'*10,'Exe %s ( %d / %d )'%(db_time_key,cnt_db_time+1,len(cand_lst)))


        db_feats = []
        db_imgpaths = []
        db_attrs = []
        for v in n_db_times[db_time_key]:
            db_time_full = pj(datapath, v)
            [db_feat, db_imgpath, db_attr] = load_obj(db_time_full)
            assert db_feat.shape[0] == len(db_imgpath) and len(db_imgpath) == len(db_attr) , 'ERR SHAPE'
            db_feats.append(db_feat)
            db_imgpaths += db_imgpath
            db_attrs += db_attr

        # db_attr = np.array(db_attr)
        db_imgpaths = np.array(db_imgpaths)

        #normalize unit
        db_feats = np.concatenate(db_feats,axis=0)
        db_feats = normalize(db_feats,axis=1)

        if db_feats.shape[0] > max_len:
            db_feats = db_feats[:max_len]
        if db_imgpaths.shape[0] > max_len:
            db_imgpaths = db_imgpaths[:max_len]
        if len(db_attrs) > max_len:
            db_attrs = db_attrs[:max_len]

        save_time_full = pj(savepath,db_time_key.replace('.pkl',''))
        if not os.path.isdir(pj(save_time_full)):
            os.mkdir(save_time_full)

        for _attr in cluster_attr:
            sub_id = [ 1 if c.startswith(_attr) and c not in exclude_attr else 0 for c in db_attrs ]
            sub_id = np.array(sub_id)

            save_attr_full = pj(save_time_full,_attr)
            if sub_id.sum() > 0:
                print('#' * 5, 'clustering for %s' % _attr)
                if not os.path.isdir(save_attr_full):
                    os.mkdir(save_attr_full)
            else:
                print('#'*5,'skip for %s'%_attr)
                continue

            #sub_xxx: data in one Floor
            sub_id = sub_id == 1

            sub_db_feat = db_feats[sub_id]
            sub_db_imgpath = db_imgpaths[sub_id]

            print('%d to be merged'%sub_db_feat.shape[0])

            model = AgglomerativeClustering(linkage=args.linkage,distance_threshold=args.threshold,n_clusters=None)
            model = model.fit(sub_db_feat)

            label = model.labels_
            n_cluster = model.n_clusters_
            # print('%d clusters'%n_cluster)

            real_cluster = 0
            for i in range(n_cluster):
                save_cluster_full = pj(save_attr_full,'%d'%i)
                if not os.path.isdir(save_cluster_full):
                    os.mkdir(save_cluster_full)
                c_label_id = label == i
                c_db_imgpath = sub_db_imgpath[c_label_id]

                #find top5
                FLG_save_cluster = True
                if args.gallery is not None and gfeat.shape[0] > args.top:
                    i_cls_feat = sub_db_feat[c_label_id]
                    i_cls_feat = i_cls_feat.mean(axis=0)
                    pgMtx = i_cls_feat@gfeat.transpose()
                    pgMtx = pgMtx.reshape(-1)
                    topid = largest_indices(pgMtx,args.top)[0]

                    if pgMtx[topid[0]] > args.topsim:
                        FLG_save_cluster = True
                    else:
                        FLG_save_cluster = False

                    if FLG_save_cluster:
                        save_cluster_candidant_full = pj(save_cluster_full, 'top%d' % args.top)
                        if not os.path.isdir(save_cluster_candidant_full):
                            os.mkdir(save_cluster_candidant_full)
                        for id in topid:
                            vs = gpath[id].split('/')
                            img_name = vs[-1] if vs[-1] != '' else vs[-2]
                            img_name = '%0.3f_%s'%(pgMtx[id],img_name)
                            shutil.copyfile(gpath[id] , pj(save_cluster_candidant_full, img_name))


                if FLG_save_cluster:
                    real_cluster +=1
                    print('%d-th cluster' %real_cluster, end='\r')
                    # moving files
                    for v in c_db_imgpath:
                        vs = v.split('/')
                        img_name = vs[-1] if vs[-1] != '' else vs[-2]
                        shutil.copyfile(v, pj(save_cluster_full, img_name))
                else:
                    os.rmdir(save_cluster_full)

            print('real_cluster:%d' % real_cluster)

    print('#'*10,'END','#'*10)






