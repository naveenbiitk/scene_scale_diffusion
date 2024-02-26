import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.carla_dataset import *

dataset_choices = {'carla', 'kitti'}


def get_data_id(args):
    return '{}'.format(args.dataset)

def get_class_weights(freq):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(freq + epsilon_w))

    return weights

def get_data(args):
    assert args.dataset in dataset_choices
    if args.dataset == 'carla':
        train_dir = "/nethome/nnagarathinam6/diffusion_ws/diffusion_data/Cartesian/Train"
        val_dir = "/nethome/nnagarathinam6/diffusion_ws/diffusion_data/Cartesian/Val"
        test_dir = "/nethome/nnagarathinam6/diffusion_ws/diffusion_data/Cartesian/Test"

        x_dim = 128
        y_dim = 128
        z_dim = 8
        data_shape = [x_dim, y_dim, z_dim]
        args.data_shape= data_shape

        binary_counts = True
        transform_pose = True
        remap = True
        if remap:
            class_frequencies = remap_frequencies_cartesian
            args.num_classes = 11
        else:
            args.num_classes = 23

        num_classes = args.num_classes
        comp_weights = get_class_weights(class_frequencies).to(torch.float32)
        seg_weights = get_class_weights(class_frequencies[1:]).to(torch.float32)
        

        
        train_ds = CarlaDataset(directory=train_dir, random_flips=True, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
        val_ds = CarlaDataset(directory=val_dir, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
        test_ds = CarlaDataset(directory=test_dir, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)

        coor_ranges = train_ds._eval_param['min_bound'] + train_ds._eval_param['max_bound']
        voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
                    abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
                    abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV
        
        if args is not None and args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, shuffle=False)
            train_iters = len(train_sampler) // args.batch_size
            val_iters = len(val_sampler) // args.batch_size
            test_iters = len(test_sampler) // args.batch_size
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
            train_iters = len(train_ds) // args.batch_size
            val_iters = len(val_ds) // args.batch_size
            test_iters = len(test_ds) // args.batch_size
        
        dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=train_ds.collate_fn, num_workers=args.num_workers)
        dataloader_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, collate_fn=val_ds.collate_fn, num_workers=args.num_workers)
        dataloader_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, sampler=test_sampler, collate_fn=test_ds.collate_fn, num_workers=args.num_workers)   # error here
    
        # b_c = 0
        # for iterate, (voxel_input, output, counts) in enumerate(dataloader_test):
        #     print(f"Iteration: {iterate}")
        #     print(f"voxel_input shape: {len(voxel_input)}")
        #     print(f"output shape: {len(output)}")
        #     print(f"counts shape: {counts}")
    
    if args.dataset == 'kitti':
        train_dir = "/nethome/nnagarathinam6/diffusion_ws/diffusion_data/KITTI/Train"
        val_dir = "/nethome/nnagarathinam6/diffusion_ws/diffusion_data/KITTI/Val"
        test_dir = "/nethome/nnagarathinam6/diffusion_ws/diffusion_data/KITTI/Test"

        x_dim = 256
        y_dim = 256
        z_dim = 32
        data_shape = [x_dim, y_dim, z_dim]
        args.data_shape= data_shape

        binary_counts = True
        transform_pose = True
        remap = True
        num_classes = 20
        cylindrical = False
        class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
        

        comp_weights = get_class_weights(class_frequencies).to(torch.float32)
        seg_weights = get_class_weights(class_frequencies[1:]).to(torch.float32)

        coor_ranges = [0,-25.6,-2] + [51.2,25.6,4.4]
        voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
              abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
              abs(coor_ranges[5] - coor_ranges[2]) / z_dim] 


        # Data Loaders
        train_ds = KittiDataset(directory=train_dir, device=device, num_frames=T, random_flips=True, remap=remap, split='train', binary_counts=binary_counts, transform_pose=transform_pose)
        val_ds = KittiDataset(directory=val_dir, device=device, num_frames=T, remap=remap, split='valid', binary_counts=binary_counts, transform_pose=transform_pose)
        test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap)
        
        B = args.batch_size
        num_workers = args.num_workers
        dataloader = DataLoader(train_ds, batch_size=B, shuffle=True, collate_fn=carla_ds.collate_fn, num_workers=num_workers)
        dataloader_val = DataLoader(val_ds, batch_size=B, shuffle=True, collate_fn=val_ds.collate_fn, num_workers=num_workers)
        dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)




    return dataloader, dataloader_val, dataloader_test, num_classes, comp_weights, seg_weights, train_sampler
