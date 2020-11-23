import torch
import os
import numpy as np


class CheckpointMgr(object):
    
    def __init__(self, ckpt_dir, max_remain=3):
        super(CheckpointMgr, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.max_remain = max_remain
        self.ckpt_save_fn_list = []
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
    
    def get_latest_checkpoint(self):
        ckpt_dir = self.ckpt_dir
        ckpt_fpath_list = [os.path.join(ckpt_dir, fn) for fn in os.listdir(ckpt_dir)
                           if (os.path.isfile(os.path.join(ckpt_dir, fn)) and fn.endswith('.pth'))]
        if len(ckpt_fpath_list) == 0:
            return None
        else:
            modify_time_list = [os.path.getmtime(fpath) for fpath in ckpt_fpath_list]
            latest_fpath_idx = np.argmax(modify_time_list)
            latest_ckpt_fpath = ckpt_fpath_list[int(latest_fpath_idx)]
            print('latest_ckpt_fpath: ', latest_ckpt_fpath)
            return latest_ckpt_fpath
    
    def load_checkpoint(self, model, ckpt_fpath=None, warm_load=False, map_location=None):
        ckpt_fpath = ckpt_fpath if ckpt_fpath is not None else self.get_latest_checkpoint()
        if ckpt_fpath is None:
            print('None ckpt file can be used, load fail.')
            return None
        if not warm_load:
            save_dict = torch.load(ckpt_fpath, map_location=map_location)
            save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
            save_dict = {key.replace('module.', ''): val
                         for key, val in save_dict.items()}
            model.load_state_dict(save_dict)
            return None
        else:
            warm_load_params = self._warm_load_weights(model, ckpt_fpath)
            return warm_load_params

    def _warm_load_weights(self, model, ckpt_path):
        model_dict = model.state_dict()
        save_dict = torch.load(ckpt_path)
        save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
        load_dict = {}
        new_params = {}
        for k, v in model_dict.items():
            if k not in save_dict.keys():
                print('warm_load_weights/new_val: ', k, v.size())
                new_params[k] = v
            elif save_dict[k].size() != v.size():
                print('warm_load_weights/change_val: {}, {}-->{}'.format(k, save_dict[k].size(), v.size()))
                new_params[k] = v
            else:
                load_dict.setdefault(k, save_dict.get(k))
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)
        return new_params
    
    def save_checkpoint(self, model, ckpt_fpath=None):
        if ckpt_fpath is not None:
            if not os.path.isabs(ckpt_fpath):
                ckpt_fpath = os.path.join(self.ckpt_dir, ckpt_fpath)
        else:
            # ckpt_fpath = None
            if len(self.ckpt_save_fn_list) > 0:
                prev_fn = self.ckpt_save_fn_list[-1]
            else:
                prev_fn = self.get_latest_checkpoint()
                prev_fn = prev_fn if prev_fn is None else prev_fn.split('/')[-1]
            if prev_fn is None or not prev_fn.startswith('model_'):
                save_fn = 'model_0.pth'
            else:
                save_fn = 'model_{}.pth'.format(int(prev_fn.replace('.pth', '').split('_')[-1]) + 1)
            ckpt_fpath = os.path.join(self.ckpt_dir, save_fn)
            self.ckpt_save_fn_list.append(save_fn)
            if len(self.ckpt_save_fn_list) > self.max_remain:
                delete_fn = self.ckpt_save_fn_list[0]
                self.ckpt_save_fn_list.remove(delete_fn)
                os.remove(os.path.join(self.ckpt_dir, delete_fn))
        print('save_ckpt_fpath: ', ckpt_fpath)
        torch.save(model.state_dict(), ckpt_fpath)
        return ckpt_fpath




