import torch
import torch.nn as nn

class PrimitiveLoss(nn.Module):

    def __init__(self, config):
        super(PrimitiveLoss, self).__init__()
        self.scale = config.scale_primitive_loss

    def forward(self, primitive_sdf):
        primitive_loss = torch.mean((primitive_sdf.min(dim=1)[0])**2) * self.scale
        return primitive_loss

class ReconLoss(nn.Module):

    def __init__(self, config):
        super(ReconLoss, self).__init__()

    def forward(self, pred_point_value, gt_point_value):
        loss_recon = torch.mean((pred_point_value - gt_point_value)**2)
        return loss_recon

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.primitive_loss = PrimitiveLoss(config)
        self.recon_loss = ReconLoss(config)

    def forward(self, predict_occupancy, gt_occupancy, primitive_sdf):
        loss_recon = self.recon_loss(predict_occupancy, gt_occupancy)
        loss_primitive = self.primitive_loss(primitive_sdf)
        loss_total = loss_recon + loss_primitive
        return {"loss_recon":loss_recon, "loss_primitive":loss_primitive, "loss_total":loss_total}








