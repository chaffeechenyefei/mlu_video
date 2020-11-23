
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank==0:
        tensor += 1
        req = dist.send(tensor=tensor, dst=1)
    else:
        req = dist.recv(tensor=tensor, src=0)
    req.wait()
    print('Rank:{} has data {}'.format(rank, tensor[0]))


def init_process(rank, size, fn, backend='nccl'):

    os.environ['MASTER_ADDR'] = '117.50.25.179'
    os.environ['MASTER_PORT'] = '3389'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def main():
    size=2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__=='__main__':

    main()





