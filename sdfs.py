import torch
import torch.nn as nn
import torch.nn.functional as F

# quaternion code are copied from pytorch3d
def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a, b):
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quaternion_invert(quaternion):
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternion_apply(quaternion, point):
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def transform_points(quaternion, translation, points):
    quaternion = nn.functional.normalize(quaternion, dim=-1)
    transformed_points = points.unsqueeze(2) - translation.unsqueeze(1)
    transformed_points = quaternion_apply(quaternion.unsqueeze(1), transformed_points)
    return transformed_points

def sdfPlane(quaternion, translation, offset, points):
    transformed_points = transform_points(quaternion, translation, points)
    distance = transformed_points[:,:,:,2].unsqueeze(-1) - offset.unsqueeze(1)
    return distance

def sdfCylinder(quaternion, translation, radius, points):
    transformed_points = transform_points(quaternion, translation, points)
    radius = torch.abs(radius)
    px = transformed_points[:,:,:,0]
    py = transformed_points[:,:,:,1]
    pz = transformed_points[:,:,:,2]
    distance = torch.norm(torch.stack((px, py), dim=-1), dim=-1)
    distance = distance.unsqueeze(-1) - radius.unsqueeze(1)
    return distance

def sdfBox(quaternion, translation, dims, points):
    B,N,_ = points.shape
    _,K,_ = quaternion.shape
    dims = torch.abs(dims)
    transformed_points = transform_points(quaternion, translation, points)
    q_points = transformed_points.abs() - dims.unsqueeze(1).repeat(1,N,1,1)
    lengths = (q_points.max(torch.zeros_like(q_points))).norm(dim=-1)
    zeros_points = torch.zeros_like(lengths)
    xs = q_points[..., 0]
    ys = q_points[..., 1]
    zs = q_points[..., 2]
    filling = ys.max(zs).max(xs).min(zeros_points)
    return lengths + filling

def sdfSphere(quaternion, translation, radius, points):
    radius = torch.abs(radius)
    transformed_points = transform_points(quaternion, translation, points)
    distance = transformed_points.norm(dim=-1).unsqueeze(-1) - radius.unsqueeze(1)
    return distance

def sdfCone(quaternion, translation, tan_alpha, points):
    tan_alpha = torch.abs(tan_alpha)
    transformed_points = transform_points(quaternion, translation, points)
    distance_to_apex = torch.norm(transformed_points, dim=-1).unsqueeze(-1)
    px = transformed_points[:,:,:,0]
    py = transformed_points[:,:,:,1]
    pz = transformed_points[:,:,:,2]
    distance_1 = torch.norm(torch.stack((px, py), dim=-1),dim=-1).unsqueeze(-1) - pz.unsqueeze(-1) * tan_alpha.unsqueeze(1).repeat(1,points.shape[1],1,1)
    cos_alpha = torch.div(1,torch.sqrt(1+ tan_alpha**2))
    distance_to_surface = distance_1 * cos_alpha.unsqueeze(1).repeat(1,points.shape[1],1,1)
    signed_distance = torch.where(pz.unsqueeze(-1) < 0, distance_to_apex, distance_to_surface)
    return signed_distance

if __name__ == "__main__":
    B=2
    N=10
    K=4
    points = torch.randn([B,N,3])
    quaternion =torch.randn([B,K,4])
    translation = torch.randn([B,K,3])
    dim = torch.randn([B,K,1]).repeat(1,1,3)
    box_sdf = sdfBox(quaternion, translation, points, dim)

