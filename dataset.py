import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)

from torch.utils.data import Dataset
from tqdm import tqdm
import os
import glob
import random
import open3d as o3d

category_ids = {
    'Airplane': '02691156',
    'Car': '02958343',
    'Chair': '03001627',
    'Lamp': '03636649',
    'Table': '04379243',
    'Sofa':'04256520',
    'Telephone': '04401088',
    'Vessel':'04530566',
    'Loudspeaker':'03691459',
    'Cabinet': '02933112',
    'Display':'03211117',
    'Bench':'02828884',
    'Rifle':'04090263'
    }

avaliable_classes = ["03001627",
                    "02958343",
                    "04256520",
                    "02691156",
                    "03636649",
                    "04401088",
                    "04530566",
                    "03691459",
                    "02933112",
                    "04379243",
                    "03211117",
                    "02828884",
                    "04090263"]

class ShapeNet(Dataset):

    def __init__(self, partition="train", category="02691156", shapenet_root="./data/ShapeNet", num_surface_points=2048, num_sample_points=2048, balance=True):
        super().__init__()
        self.balance = balance
        self.shapenet_root = shapenet_root
        self.category = category
        self.partition = partition
        self.num_surface_points = num_surface_points
        self.num_sample_points = num_sample_points
        self.surface_files, self.sample_files = self.__get_shapenet_files_category__(self.category, split=self.partition)

    def __get_shapenet_files_category__(self, category, split="train"):
        print("Category: %s Split: %s" % (category, split))
        print("%s/%s/%s.lst" % (self.shapenet_root, category, split))
        if split in ["train", "test", "val"]:
            with open("%s/%s/%s.lst" % (self.shapenet_root, category, split), "r") as f:
                files = ["%s/%s/%s" % (self.shapenet_root, category, line.strip('\n')) for line in f.readlines()] 
        else:
            print("Errror, no split named: %s. Only train, test and val are supported..." % split)
            exit(0)
        surfaces_file_paths = [i+"/pointcloud.npz" for i in files]
        samples_file_paths = [i+"/points.npz" for i in files]
        return surfaces_file_paths, samples_file_paths

    def __getitem__(self, item):
        '''
        :param item: int
        :return: surface points [N, 3]
        :return: sdf testing points [M, 4]
        '''
        # Setting up file path
        surface_file = self.surface_files[item]
        sample_file = self.sample_files[item]

        # Getting Surface Points
        surface_pointcloud = np.load(surface_file)["points"]
        surface_selection_index = np.random.randint(0, surface_pointcloud.shape[0], self.num_surface_points)
        surface_pointcloud = surface_pointcloud[surface_selection_index].astype(np.float32)

        # Getting Sample Points
        samples = np.load(sample_file)
        sample_coordinates = samples["points"]
        occupancies = np.unpackbits(samples["occupancies"])
        samples = np.concatenate([sample_coordinates, occupancies.reshape(-1,1)], axis=-1)


        if self.balance:
            # sample equal number of point from inside and outside
            inner_points = samples[samples[:,-1]==1]
            outer_points = samples[samples[:,-1]==0]
            inner_index = np.random.randint(0, inner_points.shape[0], self.num_sample_points//2)
            outer_index = np.random.randint(0, outer_points.shape[0], self.num_sample_points//2)
            samples = np.concatenate([inner_points[inner_index], outer_points[outer_index]], axis=0)
            np.random.shuffle(samples)
        else:
            # random sample points from all testing points
            sample_index = np.random.randint(0, samples.shape[0], self.num_sample_points)
            samples = samples[sample_index]
            np.random.shuffle(samples)

        return surface_pointcloud.astype(np.float32), samples.astype(np.float32)

    def __getitem__sg2_(self, item):
        '''
        :param item: int
        :return: surface points [N, 3]
        :return: sdf testing points [M, 4]
        '''
        # Setting up file path
        surface_file = self.surface_files[item]
        sample_file = self.sample_files[item]

        # Getting Surface Points
        surface_pointcloud = np.load(surface_file)["points"]
        surface_selection_index = np.random.randint(0, surface_pointcloud.shape[0], self.num_surface_points)
        surface_pointcloud = surface_pointcloud[surface_selection_index].astype(np.float32)

        # Getting Sample Points
        samples = np.load(sample_file)
        sample_coordinates = samples["points"]
        occupancies = np.unpackbits(samples["occupancies"])
        samples = np.concatenate([sample_coordinates, occupancies.reshape(-1,1)], axis=-1)

        if self.balance:
            # sample equal number of point from inside and outside
            inner_points = samples[samples[:,-1]==1]
            outer_points = samples[samples[:,-1]==0]
            inner_index = np.random.randint(0, inner_points.shape[0], self.num_sample_points//2)
            outer_index = np.random.randint(0, outer_points.shape[0], self.num_sample_points//2)
            samples = np.concatenate([inner_points[inner_index], outer_points[outer_index]], axis=0)
            np.random.shuffle(samples)
        else:
            # random sample points from all testing points
            sample_index = np.random.randint(0, sample_coordinates.shape[0], self.num_sample_points)
            samples = np.concatenate([sample_coordinates[sample_index], occupancies[sample_index]], axis=0)
            np.random.shuffle(samples)

        return surface_pointcloud.astype(np.float32), samples.astype(np.float32)

    def __len__(self):
        return len(self.sample_files)