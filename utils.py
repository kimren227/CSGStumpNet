import numpy as np
np.random.seed(233)
import torch
torch.manual_seed(233)
import os
import pathlib
import mcubes
import open3d as o3d
from tqdm import tqdm
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


def generate_mesh(model, surface_point_cloud, config, test_iter, iso_value=0.5):
    feature = model.encoder(surface_point_cloud)
    code = model.decoder(feature)
    intersection_layer_connections, union_layer_connections = model.connection_head(code, is_training=False)
    primitive_parameters = model.primitive_head(code)
    occ_func = lambda sample_points: (model.csg_stump(sample_points, primitive_parameters, intersection_layer_connections, union_layer_connections, is_training=False)[0]).detach().cpu().numpy()
    mc = MarchingCubes(config.real_size, config.test_size, use_pytorch=True)
    file_prefix = os.path.join(*[config.sample_dir, config.experiment_name])
    mc.batch_export_mesh(file_prefix, test_iter*surface_point_cloud.shape[0], surface_point_cloud.shape[0], occ_func, iso_value)

# def gen_results(model, surface, config, test_iter):
#     # chunk the high res voxel to smaller chunk
#     test_point_batch_size = config.test_size * config.test_size * config.test_size  # do not change
#     dima = config.test_size
#     dim = config.real_size
#     aux_x = np.zeros([dima, dima, dima], np.uint8)
#     aux_y = np.zeros([dima, dima, dima], np.uint8)
#     aux_z = np.zeros([dima, dima, dima], np.uint8)
#     multiplier = int(dim / dima)
#     multiplier2 = multiplier * multiplier
#     multiplier3 = multiplier * multiplier * multiplier
#     for i in range(dima):
#         for j in range(dima):
#             for k in range(dima):
#                 aux_x[i, j, k] = i * multiplier
#                 aux_y[i, j, k] = j * multiplier
#                 aux_z[i, j, k] = k * multiplier
#     coords = np.zeros([multiplier3, dima, dima, dima, 3], np.float32)
#     for i in range(multiplier):
#         for j in range(multiplier):
#             for k in range(multiplier):
#                 coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = aux_x + i
#                 coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = aux_y + j
#                 coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = aux_z + k



#     coords = (coords + 0.5) / dim - 0.5
#     coords = np.reshape(coords, [multiplier3, test_point_batch_size, 3])  # 8,32*32*32,3
#     coords = torch.from_numpy(coords)
#     z_vector, out_m, _, _, raw_sdf = model(surface.transpose(2,1), None, None, None, is_training=False)

#     batch_size = out_m.shape[0]


#     model_float = np.zeros([batch_size, config.real_size + 2, config.real_size + 2, config.real_size + 2], np.float32)
#     model_parts = []
#     model_primitives = []
#     # for i in range(config.c_dim):
#     #     model_parts.append(np.zeros([batch_size, config.real_size + 2, config.real_size + 2, config.real_size + 2], np.float32))

#     # for i in range(config.p_dim):
#     #     model_primitives.append(np.zeros([batch_size, config.real_size + 2, config.real_size + 2, config.real_size + 2], np.float32))

#     pointclouds = []
#     occ_pred = []
#     with tqdm(total=multiplier**3) as pbar:
#         for i in range(multiplier):
#             for j in range(multiplier):
#                 for k in range(multiplier):
#                     minib = i * multiplier2 + j * multiplier + k
#                     point_coord = coords[minib:minib + 1]
#                     batch_point_cood = point_coord.repeat(batch_size,1,1)
#                     _, _, parts, model_out, raw_sdf = model(None, z_vector, out_m, batch_point_cood, is_training=False)
#                     pointclouds.append(batch_point_cood.cpu().detach().numpy())
#                     occ_pred.append(model_out.cpu().detach().numpy())
#                     model_float[:,aux_x + i + 1, aux_y + j + 1, aux_z + k + 1] = np.reshape(model_out.squeeze(-1).squeeze(-1).detach().cpu().numpy(), [batch_size, config.test_size, config.test_size, config.test_size])
#                     # for ii in range(config.c_dim):
#                     #     model_parts[ii][:,aux_x + i + 1, aux_y + j + 1, aux_z + k + 1] = np.reshape(parts[...,ii].squeeze(-1).squeeze(-1).detach().cpu().numpy(), [batch_size, config.test_size, config.test_size, config.test_size])
#                     # for ii in range(config.p_dim):
#                     #     model_primitives[ii][:,aux_x + i + 1, aux_y + j + 1, aux_z + k + 1] = np.reshape(raw_sdf[...,ii].squeeze(-1).squeeze(-1).detach().cpu().numpy(), [batch_size, config.test_size, config.test_size, config.test_size])
                    
#                     pbar.update(1)

#     if not os.path.exists(config.sample_dir):
#         os.mkdir(config.sample_dir)
#     if not os.path.exists(os.path.join(config.sample_dir, config.experiment_name)):
#         os.mkdir(os.path.join(config.sample_dir, config.experiment_name))

#     config.csg_dir = "./csg/"
#     if not os.path.exists(config.csg_dir):
#         os.mkdir(config.csg_dir)
#     if not os.path.exists(os.path.join(config.csg_dir, config.experiment_name)):
#         os.mkdir(os.path.join(config.csg_dir, config.experiment_name))

#     for i in range(batch_size):
#         npy_path = os.path.join(*[config.csg_dir, config.experiment_name, str(test_iter*batch_size+i) + ".npy"])
#         cvx_weights = model.module.cvx_weights.cpu().detach().numpy()
#         ccv_weights = model.module.ccv_weights.cpu().detach().numpy()
#         plane_m_save = out_m.cpu().detach().numpy()
#         np.save(npy_path, {"cvx": cvx_weights[i], "ccv":ccv_weights[i], "primitive":plane_m_save[i]}) 
#         # exit(0)
#         vertices, triangles = mcubes.marching_cubes(model_float[i], 0.5) # original mesh without smoothing
#         #vertices, triangles = mcubes.marching_cubes(1-model_float[i], 0.5) # original mesh without smoothing
#         # vertices, triangles = mcubes.marching_cubes(mcubes.smooth((model_float[i]<0).astype(np.float32)), 0.5)
#         vertices = (vertices - 0.5) / (config.real_size) - 0.5
#         mesh = o3d.geometry.TriangleMesh()
#         mesh.vertices = o3d.utility.Vector3dVector(vertices)
#         mesh.triangles = o3d.utility.Vector3iVector(triangles)
#         o3d.io.write_triangle_mesh(os.path.join(*[config.sample_dir, config.experiment_name, str(test_iter*batch_size+i) + "_vox.ply"]), mesh)
#         print("Vertice Count: %d Triangle Count: %d" % (vertices.shape[0], triangles.shape[0]))
#         # for j in range(config.c_dim):
#         #     # vertices, triangles = mcubes.marching_cubes(mcubes.smooth((model_parts[j][i]<0).astype(np.float32)), 0.5)
#         #     vertices, triangles = mcubes.marching_cubes(model_parts[j][i]*-1, 0)

#         #     vertices = (vertices - 0.5) / (config.real_size) - 0.5
#         #     mesh = o3d.geometry.TriangleMesh()
#         #     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#         #     mesh.triangles = o3d.utility.Vector3iVector(triangles)
#         #     o3d.io.write_triangle_mesh(os.path.join(*[config.sample_dir, config.experiment_name, str(test_iter*batch_size+i) + "_%d_part.ply" % j]), mesh)
#         #     print("Part %d Vertice Count: %d Triangle Count: %d" % (j, vertices.shape[0], triangles.shape[0]))
#         # for j in range(config.p_dim):
#         #     # vertices, triangles = mcubes.marching_cubes(mcubes.smooth((model_parts[j][i]<0).astype(np.float32)), 0.5)
#         #     vertices, triangles = mcubes.marching_cubes(model_primitives[j][i]*-1, 0)

#         #     vertices = (vertices - 0.5) / (config.real_size) - 0.5
#         #     mesh = o3d.geometry.TriangleMesh()
#         #     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#         #     mesh.triangles = o3d.utility.Vector3iVector(triangles)
#         #     o3d.io.write_triangle_mesh(os.path.join(*[config.sample_dir, config.experiment_name, str(test_iter*batch_size+i) + "_%d_primitives.ply" % j]), mesh)
#         #     print("Primitive %d Vertice Count: %d Triangle Count: %d" % (j, vertices.shape[0], triangles.shape[0]))
#         # exit(0)

