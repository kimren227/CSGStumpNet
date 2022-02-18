import numpy as np
np.random.seed(233)
import torch
torch.manual_seed(233)
import os
import pathlib
from marchingcube import MarchingCubes

def init(config):
    if not os.path.exists('./checkpoints/%s/models' % config.experiment_name):
        pathlib.Path('./checkpoints/%s/models' % config.experiment_name).mkdir(parents=True, exist_ok=True)
    if not os.path.exists('./checkpoints/%s/code' % config.experiment_name):
        pathlib.Path('./checkpoints/%s/code' % config.experiment_name).mkdir(parents=True, exist_ok=True)
    if not os.path.exists('./%s/%s' % (config.sample_dir, config.experiment_name)):
        pathlib.Path('./%s/%s' % (config.sample_dir, config.experiment_name)).mkdir(parents=True, exist_ok=True)
    if not os.path.exists('./%s/%s' % (config.csg_dir, config.experiment_name)):
        pathlib.Path('./%s/%s' % (config.csg_dir, config.experiment_name)).mkdir(parents=True, exist_ok=True)
    os.system("cp *.py ./checkpoints/%s/code" % config.experiment_name) 

def point_inside_box(points, pointcloud):
    bbox_min = pointcloud.transpose(2,1).min(dim=-2)[0]
    bbox_max = pointcloud.transpose(2,1).max(dim=-2)[0]
    inside = ((points > bbox_max.unsqueeze(1)).sum(dim=-1) + (points < bbox_min.unsqueeze(1)).sum(dim=-1)) == 0
    return inside

def generate_mesh(model, surface_point_cloud, config, test_iter, iso_value=0.5):
    feature = model.encoder(surface_point_cloud)
    code = model.decoder(feature)
    intersection_layer_connections, union_layer_connections = model.connection_head(code, is_training=False)
    primitive_parameters = model.primitive_head(code)
    occ_func = lambda sample_points: (model.csg_stump(sample_points, primitive_parameters, intersection_layer_connections, union_layer_connections, is_training=False)[0]).detach().cpu().numpy()
    padded_occ_func = lambda sample_points: occ_func(sample_points) * point_inside_box(sample_points, surface_point_cloud).detach().cpu().numpy()
    mc = MarchingCubes(config.real_size, config.test_size, use_pytorch=True)
    file_prefix = os.path.join(*[config.sample_dir, config.experiment_name])
    mc.batch_export_mesh(file_prefix, test_iter*surface_point_cloud.shape[0], surface_point_cloud.shape[0], padded_occ_func, iso_value)
