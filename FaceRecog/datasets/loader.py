import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets.prefetcher_io import DataLoaderX
from torch.utils.data.distributed import DistributedSampler

class FaceDataLoader(object):

    def __init__(self,
                 imgs_per_gpu,
                 workers_per_gpu,
                 num_gpus=1,
                 dist=False,
                 shuffle=True,
                 pin_memory=True,
                 **kwargs):
        super(FaceDataLoader, self).__init__()
        print('################### Init FaceDataLoader. ###################')
        self.imgs_per_gpu = imgs_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.num_gpus = num_gpus
        self.dist = dist
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.batch_size = imgs_per_gpu * num_gpus
        self.num_workers = workers_per_gpu * num_gpus

    def __call__(self, dataset):
        if not self.dist:
            return DataLoaderX(dataset=dataset,
                              batch_size=self.batch_size,
                              shuffle=self.shuffle,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=True)
        else:
            return DataLoader(dataset=dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              sampler=DistributedSampler(dataset),
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=True)


if __name__ == '__main__':

    from datasets.faces import FaceDataset
    dataset = FaceDataset()
    data_loader = FaceDataLoader(imgs_per_gpu=2,
                                 workers_per_gpu=1,
                                 num_gpus=1)(dataset)

    for i, data_batch in enumerate(data_loader):
        print('------ batch_idx: {} ------'.format(i))
        imgs, labels = data_batch
        print(imgs.size())
        print(labels)
