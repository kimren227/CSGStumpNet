from config import Config
import argparse
import open3d as o3d
import numpy as np
import os
import torch
import glob
from tqdm import tqdm
from dataset import ShapeNet
from chamfer_distance import ChamferDistance

def chamfer_distance(source_cloud, target_cloud):
    source_cloud = torch.tensor(source_cloud).unsqueeze(0).cuda()
    target_cloud = torch.tensor(target_cloud).unsqueeze(0).cuda()
    chamferDist = ChamferDistance()
    distance_1, distance_2 = chamferDist(source_cloud, target_cloud)
    distance_1 = distance_1.mean()
    distance_2 = distance_2.mean()
    return distance_1.item(), distance_2.item()

def get_chamfer_distance(gt_pointcloud, output_mesh):
    mesh = o3d.io.read_triangle_mesh(output_mesh)
    pcd = mesh.sample_points_poisson_disk(2048)
    pred_points = np.asarray(pcd.points, dtype=np.float32)
    distance = chamfer_distance(gt_pointcloud, pred_points)
    return distance

def get_all_mesh_indices(mesh_dir):
    mesh_path = glob.glob(mesh_dir+"/*.ply")
    return [int(path.split("/")[-1].split(".")[0]) for path in mesh_path]

def calc_chamfer(config):
    dataset = ShapeNet(partition='test', category=config.category, shapenet_root=config.dataset_root, balance=config.balance,num_surface_points=config.num_surface_points, num_sample_points=config.num_sample_points)
    samples_dir = os.path.join(config.sample_dir, config.experiment_name)
    d1s = []
    d2s = []
    for index in tqdm(get_all_mesh_indices(samples_dir)):
        pointcloud, _ = dataset[index]
        pred_mesh = samples_dir+"/%d.ply" % index
        d1, d2 = get_chamfer_distance(pointcloud, pred_mesh)
        d1s.append(d1)
        d2s.append(d2)
    return sum(d1s)/len(d1s), sum(d2s)/len(d2s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSGStumpNet')
    parser.add_argument('--config_path', type=str, default='./configs/config_default.json', metavar='N',
                        help='config_path')
    args = parser.parse_args()
    config = Config((args.config_path))
    cd1, cd2 = calc_chamfer(config)
    print("")
    print("=======%s========" % config.experiment_name)
    print("Without Scale:")
    print("CD: %f %f, average: %f" % (cd1, cd2, (cd1+cd2)/2))
    print("Scale by 1000:")
    print("CD: %f %f, average: %f" % (cd1*1000, cd2*1000, (cd1+cd2)*500))

