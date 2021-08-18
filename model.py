import torch
import torch.nn as nn
import torch.nn.functional as F

from sdfs import *
from dgcnn import DGCNNFeat

class CSGStump(nn.Module):
    def __init__(self, num_primitives, num_intersections, sharpness):
        super(CSGStump, self).__init__()
        self.num_primitives = num_primitives
        self.num_intersections = num_intersections
        self.sharpness = sharpness

    def forward(self, sample_point_coordinates, primitive_parameters, intersection_layer_weights, union_layer_weights, is_training):
        B = sample_point_coordinates.shape[0]
        M = sample_point_coordinates.shape[1] # number of testing points

        primitive_parameters = primitive_parameters.transpose(2,1)
        B,K,param_dim = primitive_parameters.shape

        # getting parameters for each type of primitive
        boxes = primitive_parameters[:,:,:10] # [B,K/4,10]
        cylinder = primitive_parameters[:,:,10:18] # [B,K/4,8]
        sphere = primitive_parameters[:,:,18:26] # [B,K/4,8]
        cone = primitive_parameters[:,:,26:] # [B,K/4,8]

        # compute sign distance w.r.t each primitive
        cylinder_sdf = sdfCylinder(cylinder[:,:,:4], cylinder[:,:,4:7], cylinder[:,:,7:], sample_point_coordinates[:,:,:3]).squeeze(-1) #[B,N,K]
        box_sdf = sdfBox(boxes[:,:,:4], boxes[:,:,4:7], boxes[:,:,7:], sample_point_coordinates[:,:,:3]).squeeze(-1) #[B,N,K]
        cone_sdf = sdfCone(cone[:,:,:4], cone[:,:,4:7], cone[:,:,7:], sample_point_coordinates[:,:,:3]).squeeze(-1) #[B,K]
        sphere_sdf = sdfSphere(sphere[:,:,:4], sphere[:,:,4:7], sphere[:,:,7:], sample_point_coordinates[:,:,:3]).squeeze(-1) #[B,N,K]

        # compute occupancies
        primitive_sdf = torch.cat([cylinder_sdf, box_sdf, cone_sdf, sphere_sdf], dim=-1)
        primitive_occupancies = torch.sigmoid(-1 * primitive_sdf * self.sharpness)

        # calculate intersections
        # W * occupancy + (1-W) * 1,  where 1 indicates solid, i.e. solid intersect anything is equal to itself
        occupancy_pre_intersection = torch.einsum("bkc,bmk->bmkc", intersection_layer_weights, primitive_occupancies) \
                                                    + torch.einsum("bkc,bmk->bmkc", 1-intersection_layer_weights, primitive_occupancies.new_ones(primitive_occupancies.shape))
        if not is_training:
            intersection_node_occupancies = torch.min(occupancy_pre_intersection, dim=-2)[0]
        else:
            with torch.no_grad():
                # use soft min to distribute gradients
                weights = torch.softmax(occupancy_pre_intersection * (-20), dim=-2)
            intersection_node_occupancies = torch.sum(weights * occupancy_pre_intersection, dim=-2) # [BMC]

        # calculate union
        # W*sdf + (1-W)*(0) where 0 indicates empty, and empty union anything is equal to itself
        occupancy_pre_union = torch.einsum("bc,bmc->bmc", union_layer_weights, intersection_node_occupancies)
        if not is_training:
            occupancies = torch.max(occupancy_pre_union, dim=-1)[0]
        else:
            with torch.no_grad():
                # use soft max to distribute gradients
                weights = torch.softmax(occupancy_pre_union * (20), dim=-1)
            occupancies = torch.sum(weights  * occupancy_pre_union, dim=-1)
        return occupancies, primitive_sdf, intersection_node_occupancies
  

class CSGStumpConnectionHead(nn.Module):
    def __init__(self, feature_dim, num_primitives, num_intersections):
        super(CSGStumpConnectionHead, self).__init__()
        self.num_primitives = num_primitives
        self.num_intersections = num_intersections
        self.feature_dim = feature_dim
        self.intersection_linear = nn.Linear(self.feature_dim * 8, self.num_primitives * self.num_intersections, bias=True)
        self.union_linear = nn.Linear(self.feature_dim * 8, self.num_intersections, bias=True)

    def forward(self, feature, is_training):
        # getting intersection layer connection weights
        intersection_layer_weights = self.intersection_linear(feature)
        intersection_layer_weights = intersection_layer_weights.view(-1, self.num_primitives, self.num_intersections) # [B, num_primitives, num_intersections]

        # getting union layer connection weights
        union_layer_weights = self.union_linear(feature)
        union_layer_weights = union_layer_weights.view(-1, self.num_intersections) # [B,c_dim]

        if not is_training:
            # during inference, we use descrtize connection weights to get interpretiable CSG relations
            intersection_layer_weights = (intersection_layer_weights>0).type(torch.float32)
            union_layer_weights = (union_layer_weights>0).type(torch.float32)
        else:
            # during train, we use continues connection weights to get better gradients
            intersection_layer_weights = torch.sigmoid(intersection_layer_weights)
            union_layer_weights = torch.sigmoid(union_layer_weights)

        return intersection_layer_weights, union_layer_weights

class CSGStumpPrimitiveHead(nn.Module):

    def __init__(self, num_primitives, feature_dim):
        super(CSGStumpPrimitiveHead, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = feature_dim
        # we support 4 types of primitives, sphere, cylinder, cone, and box. 
        # Primitives are defined by Rotation, Translation and Intrinsic Parameter
        self.num_primitive_parameters_aggregated = 8+8+8+10 # Sphere (4+3+1), Cylinder (4+3+1), Cone (4+3+1), Box (4+3+3)
        self.num_type = 4

        self.primitive_linear = nn.Linear(self.feature_dim * 8, int((self.num_primitives * self.num_primitive_parameters_aggregated)/self.num_type), bias=True)
        nn.init.xavier_uniform_(self.primitive_linear.weight)
        nn.init.constant_(self.primitive_linear.bias, 0)

    def forward(self, feature):
        shapes = self.primitive_linear(feature)
        return shapes.view(-1, self.num_primitive_parameters_aggregated, int(self.num_primitives / self.num_type)) # [B,num_primitive_parameters_aggregated, num_primitives]

class Decoder(nn.Module):

    def __init__(self, feature_dim):
        super(Decoder, self).__init__()
        self.feature_dim = feature_dim
        self.linear_1 = nn.Linear(self.feature_dim, self.feature_dim * 2, bias=True)
        self.linear_2 = nn.Linear(self.feature_dim * 2, self.feature_dim * 4, bias=True)
        self.linear_3 = nn.Linear(self.feature_dim * 4, self.feature_dim * 8, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)

    def forward(self, inputs):
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)
        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)
        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)
        return l3

class CSGStumpNet(nn.Module):

    def __init__(self, config):
        super(CSGStumpNet, self).__init__()
        self.config = config

        self.num_primitives = self.config.num_primitives
        self.num_intersections = self.config.num_intersections
        self.feature_dim = self.config.feature_dim
        self.sharpness = self.config.sharpness

        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(self.feature_dim)
        self.connection_head = CSGStumpConnectionHead(self.feature_dim, self.num_primitives, self.num_intersections)
        self.primitive_head = CSGStumpPrimitiveHead(self.feature_dim, self.num_primitives)
        self.csg_stump = CSGStump(self.num_primitives, self.num_intersections, self.sharpness)

    def forward(self, surface_pointcloud, sample_coordinates, is_training=True):
        feature = self.encoder(surface_pointcloud)
        code = self.decoder(feature)
        intersection_layer_connections, union_layer_connections = self.connection_head(code, is_training=is_training)
        primitive_parameters = self.primitive_head(code)
        occupancies, primitive_sdfs, _ = self.csg_stump(sample_coordinates, primitive_parameters, intersection_layer_connections, union_layer_connections, is_training=is_training)
        return occupancies, primitive_sdfs
