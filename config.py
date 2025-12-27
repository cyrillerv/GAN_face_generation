import torch

class Config:
    dataroot = "./dataset_faces/img_align_celeba"
    dataset_size = 25000
    workers = 2
    batch_size = 128
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 100
    lr_G = 0.0002
    lr_D = 0.0001
    beta1 = 0.5
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")