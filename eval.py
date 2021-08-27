import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ShapeNet
from loss import Loss
from config import Config
from model import CSGStumpNet
from utils import generate_mesh
import argparse

def eval(config):
    test_dataset = ShapeNet(partition='test', category=config.category, shapenet_root=config.dataset_root, balance=config.balance,num_surface_points=config.num_surface_points, num_sample_points=config.num_sample_points)
    test_loader  = DataLoader(test_dataset, pin_memory=True, num_workers=20, batch_size=config.test_batch_size_per_gpu*config.num_gpu, shuffle=False, drop_last=True)

    
    device = torch.device("cuda")
    model = CSGStumpNet(config).to(device)
    pre_train_model_path = './checkpoints/%s/models/model.th' % config.experiment_name
    assert os.path.exists(pre_train_model_path), "Cannot find pre-train model for experiment: {}\nNo such a file: {}".format(config.experiment_name, pre_train_model_path)  
    model.load_state_dict(torch.load('./checkpoints/%s/models/model.th' % config.experiment_name))
    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    criterion = Loss(config)
    model.eval()
    start_time = time.time()
    test_iter = 0
    with torch.no_grad():
        testloader_t = tqdm(test_loader)
        avg_test_loss_recon = avg_test_loss_primitive = avg_test_loss = avg_test_accuracy = avg_test_recall = 0
        for surface_pointcloud, testing_points  in testloader_t:

            surface_pointcloud = surface_pointcloud.to(device)
            testing_points = testing_points.to(device)

            occupancies, primitive_sdfs = model(surface_pointcloud.transpose(2,1), testing_points[:,:,:3], is_training=False)
            loss_dict = criterion(occupancies, testing_points[:,:,-1], primitive_sdfs)

            predict_occupancies = (occupancies >=0.5).float()
            target_occupancies = (testing_points[:,:,-1] >=0.5).float()

            accuracy = torch.sum(predict_occupancies*target_occupancies)/torch.sum(target_occupancies)
            recall = torch.sum(predict_occupancies*target_occupancies)/(torch.sum(predict_occupancies)+1e-9)

            avg_test_loss_recon += loss_dict["loss_recon"].item()
            avg_test_loss_primitive += loss_dict["loss_primitive"].item()
            avg_test_loss += loss_dict["loss_total"].item()

            avg_test_accuracy += accuracy.item()
            avg_test_recall += recall.item()

            generate_mesh(model, surface_pointcloud.transpose(2,1), config, test_iter)
            test_iter += 1

        avg_test_loss_recon = avg_test_loss_recon / test_iter
        test_accuracy = avg_test_accuracy / test_iter
        test_recall = avg_test_recall / test_iter
        test_fscore = 2*test_accuracy*test_recall/(test_accuracy + test_recall + 1e-6)
        print("Evaluating: time: %4.4f, loss_total: %.6f, loss_recon: %.6f, loss_primitive: %.6f, acc: %.6f, recall: %.6f, fscore: %.6f" % ( 
                                                                                                    time.time() - start_time, 
                                                                                                    avg_test_loss/test_iter, 
                                                                                                    avg_test_loss_recon / test_iter, 
                                                                                                    avg_test_loss_primitive/test_iter, 
                                                                                                    test_accuracy, 
                                                                                                    test_recall, 
                                                                                                    test_fscore))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EvalPartAwareReconstruction')
    parser.add_argument('--config_path', type=str, default='./configs/config_default.json', metavar='N',
                        help='config_path')
    args = parser.parse_args()
    config = Config((args.config_path))
    eval(config)





