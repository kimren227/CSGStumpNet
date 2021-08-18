import numpy as np
import mcubes
import tqdm
import open3d as o3d
import torch
from pathlib import Path

class MarchingCubes:

    def __init__(self, model_resolution, eval_resolution, use_pytorch=True):
        self.model_resolution = model_resolution
        self.eval_resolution = eval_resolution
        self.use_pytorch = use_pytorch

    def generate_testing_points(self):
        points = np.indices((self.model_resolution,self.model_resolution,self.model_resolution)).T.reshape(-1,3)
        points = (points + 0.5) / self.model_resolution - 0.5
        return points

    def generate_chunked_testing_points(self):
        testing_points = np.array_split(self.generate_testing_points(), int(self.model_resolution/self.eval_resolution)**3, axis=0)
        return testing_points

    def generate_mesh(self, occupancy_function, iso_value):
        points = self.generate_chunked_testing_points()
        if self.use_pytorch:
            points = torch.tensor(points).type(torch.float32).cuda()
        occupancies = np.concatenate([occupancy_function(pts) for pts in points], axis=0)
        verts, faces = mcubes.marching_cubes(occupancies.reshape((self.model_resolution,self.model_resolution,self.model_resolution).transpose(2,1,0)), iso_value)
        verts = ((verts - 0.5) / self.model_resolution) - 0.5
        return verts, faces 

    def batch_generate_mesh(self, batch_size, occupancy_function, iso_value):
        points = self.generate_chunked_testing_points()
        points = np.repeat(np.expand_dims(points, axis=0), batch_size, axis=0)
        if self.use_pytorch:
            points = torch.tensor(points).type(torch.float32).cuda()
        occupancies = np.concatenate([occupancy_function(points[:,i,...]) for i in range(points.shape[1])], axis=1)
        occupancies = occupancies.reshape(batch_size, self.model_resolution, self.model_resolution, self.model_resolution)
        batch_verts = []
        batch_faces = []
        for i in range(occupancies.shape[0]):
            verts, faces = mcubes.marching_cubes(occupancies[i].transpose(2,1,0), iso_value)
            verts = ((verts - 0.5) / self.model_resolution) - 0.5
            batch_verts.append(verts)
            batch_faces.append(faces)
        return batch_verts, batch_faces

    def export_mesh(self, file_name, occupancy_function, iso_value):
        verts, faces = self.generate_mesh(occupancy_function, iso_value)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(file_name, mesh)

    def batch_export_mesh(self, file_name_prefix, start_index, batch_size, occupancy_function, iso_value):
        batch_verts, batch_faces = self.batch_generate_mesh(batch_size, occupancy_function, iso_value)
        Path(file_name_prefix).mkdir(parents=True, exist_ok=True)
        for i in range(len(batch_verts)):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(batch_verts[i])
            mesh.triangles = o3d.utility.Vector3iVector(batch_faces[i])
            o3d.io.write_triangle_mesh("%s/%d.ply" % (file_name_prefix, (start_index+i)), mesh)