
from __future__ import print_function, division
import argparse
import json
import os
import pdb
import sys
import time
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from reid_utils import RandomIdentitySampler, logging, RandomErasing
from reid_utils.test_utils import *
from models.AN_EvReId_model import EvReId
from data.dataloader import *
from reid_losses import TripletLoss
from torchmetrics.functional import structural_similarity_index_measure
from torch.utils.tensorboard import SummaryWriter
from e2vid_utils.utils.loading_utils import load_E2VID, get_device
from e2vid_utils.e2vid_image_utils import *


def train_model(model_ReId, model_E2VID, optimizer, scheduler, device, num_epochs=50):
    model_E2VID = model_E2VID.eval()
    start_time = time.time()
    writer = SummaryWriter()
    # model = model.double()
    dataloaders, dataset_sizes_allSample = load_train_data()
    print("data size", dataset_sizes_allSample)
    batch_count = 0
    for epoch in range(num_epochs):
        if epoch == 0:
            save_network(model_ReId, epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:

            if opt.ReId_loss == 'softmax+triplet':
                adjust_lr_triplet(optimizer, epoch)
            else:
                adjust_lr_softmax(optimizer, epoch)

            model_ReId.train(True)  # Set model to training mode
            running_loss = 0.0
            running_correct_category = 0
            # Iterate over data.
            for data in dataloaders:
                input_voxel, labels = data
                input_voxel = Variable(input_voxel.to(device))
                labels = Variable(labels.to(device))
                #rgb = Variable(rgb.cuda())
                optimizer.zero_grad()
                
                """AN + ReId model joint training"""
                [category, feature, _, AN_voxel] = model_ReId(input_voxel)
                
                """Structure Loss between input and output voxel"""
                voxel_recons_loss = voxel_img_reconst_loss(input_voxel, AN_voxel)

                """Event ReId Losses"""
                _, category_preds = torch.max(category.data, 1)

                if opt.ReId_loss == 'softmax+triplet':
                    loss_softmax = criterion_softmax(category_preds, labels)
                    loss_triplet, _, _ = criterion_triplet(feature, labels)
                    loss_reid = loss_softmax + loss_triplet

                    """softmax"""
                else:
                    loss_reid = criterion_softmax(category, labels)

                """event-2-video block"""
                if AN_voxel is not None:
                    last_states_voxel = None
                    last_states_AN_voxel = None
                    
                    """Note: we computed image reconstruction loss between GT rgb image and reconstructed image from anonymized voxel, 
                    but we do not share GT rgb data. Instead you can compute loss between GT image reconstructed from raw voxel 
                    and predicted image from anonymized voxel """

                    GT_predicted_img, states_voxel = model_E2VID(input_voxel, last_states_voxel) 

                    AN_predicted_img, states_AN_voxel = model_E2VID(AN_voxel, last_states_AN_voxel)

                    # GT_predicted_img = un_mask_filter(GT_predicted_img)
                    # GT_predicted_img = intensity_rescaler(GT_predicted_img)

                    # AN_predicted_img = un_mask_filter(AN_predicted_img)
                    # predicted_AN_img = intensity_rescaler(AN_predicted_img)

                    """Image reconstruction Loss between GT rgb image (or GT reconstructed grayscale image from raw-voxel) 
                    and image reconstructed from anonymize-voxel"""
                    
                    img_recons_loss = voxel_img_reconst_loss(GT_predicted_img, AN_predicted_img)
                    
                    """
                    #img_recons_loss = voxel_img_reconst_loss(rgb, AN_predicted_img)
                    
                    """end of e2vid"""

                writer.add_scalars(f'batch_loss', {'voxel_loss': voxel_recons_loss.item(), 'image_loss': img_recons_loss, 'reid_loss': loss_reid}, batch_count)
                print('batch loss', voxel_recons_loss.item(), img_recons_loss.item(), loss_reid.item())
                batch_count += 1

                """Total Loss"""
                loss = loss_reid + voxel_recons_loss - img_recons_loss
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item()

                category_preds = category.data.max(1)[1]
                running_correct_category += torch.sum(category_preds == labels.data)

            epoch_loss = running_loss / dataset_sizes_allSample / opt.batchsize
            epoch_acc = running_correct_category.cpu().numpy() / dataset_sizes_allSample / opt.batchsize
            print('{} Loss: {:.4f} Acc_category: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            """tensorboard"""
            #writer.add_graph(model)
            writer.add_scalar('training/loss', epoch_loss, epoch)
            writer.add_scalar('training/accuracy', epoch_acc, epoch)
            
            if epoch == 0 or epoch == 29 or epoch == 59 or epoch == 99:
                save_network(model_ReId, epoch)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    writer.close()
    return model_ReId
    
    
def voxel_img_reconst_loss(gt, pr):

    if opt.AN_loss == 'SSIM':
        loss_ = 1 - structural_similarity_index_measure(gt, pr)

    else:
        loss_ = torch.nn.MSELoss().cuda()

    return loss_


def save_network(network1, epoch_label): 
    save_filename = 'net_%s.pth'% epoch_label
    save_path1 = os.path.join('./' + opt.file_name, save_filename)
    torch.save(network1.cpu().state_dict(), save_path1)
    if torch.cuda.is_available():
        network1.cuda(0)


def adjust_lr_triplet(optimizer, ep):
    if ep < 20:
        lr = 1e-2 * (ep + 1) / 2
    elif ep < 130:
        lr = 1e-1
    elif ep < 200:
        lr = 1e-2
    elif ep < 240:
        lr = 1e-3
    elif ep < 280:
        lr = 1e-3 * (ep - 240 + 1) / 40
    elif ep < 340:
        lr = 1e-3
    for index in range(len(optimizer.param_groups)):
        if index == 0:
            optimizer.param_groups[index]['lr'] = lr * 0.1
        else:
            optimizer.param_groups[index]['lr'] = lr


def adjust_lr_softmax(optimizer, ep):
    if ep < 40:
        lr = 0.05
    elif ep < 50:
        lr = 0.01
    else:
        lr = 0.001
    for index in range(len(optimizer.param_groups)):
        if index == 0:
            optimizer.param_groups[index]['lr'] = lr * 0.1
        else:
            optimizer.param_groups[index]['lr'] = lr


def load_train_data(num_workers=8):

    """Load Train Data"""
    if opt.represent == "voxel":

        if opt.ReId_loss == 'softmax+triplet':
            cls_datasets = voxelDataset(mode='train')
            cls_loader = torch.utils.data.DataLoader(cls_datasets,
                                                     sampler=RandomIdentitySampler(cls_datasets, opt.num_instances),
                                                     batch_size=opt.batchsize,
                                                     shuffle=False, num_workers=num_workers, drop_last=True)

            dataset_sizes_allSample = len(cls_loader)

        else:
            cls_datasets = voxelDataset(mode='train')
            cls_loader = torch.utils.data.DataLoader(cls_datasets, batch_size=opt.batchsize,
                                                     shuffle=False, num_workers=num_workers, drop_last=True)

            dataset_sizes_allSample = len(cls_loader)

        print("size of all samples", dataset_sizes_allSample)

        return cls_loader, dataset_sizes_allSample


def set_optimizer(model):
    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([
                     {'params': base_params},
                     {'params': model.model.fc.parameters()},
                     {'params': model.classifier.parameters()}
                     ], lr=0.001, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    return optimizer


if __name__ == '__main__':

    """Parser"""
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--represent', default='voxel', type=str, help='representation of events for reid')
    parser.add_argument('--An_model_block', default=True, help='set True, if implement Event-voxel Anonymization Block')
    parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--name', default='AN_Event_ReId', type=str, help='output model name')
    parser.add_argument('--num_ids', default=22, type=int, help='number of identities')
    parser.add_argument('--batchsize', default=4, type=int, help='batchsize')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='training weight decay')
    parser.add_argument('--ReId_loss', default='softmax', type=str, help='choice of reid loss')
    parser.add_argument('--AN_loss', default='SSIM', type=str, help='choice of structural and image reconstruction loss for'
                                                                     'Event Anonymization: SSIM or MSE ')
    parser.add_argument('--num_instances', default=8, type=int, help='for triplet loss')
    parser.add_argument('--margin', default=4, type=float, help='triplet loss margin')
    parser.add_argument('--num_Bin', default=5, type=int, help='number of channels of spatiotemporal event-voxel')
    parser.add_argument('--e2vid_path', default='e2vid_utils/pretrained/E2VID_lightweight.pth.tar', type=str, help='path to e2vid weights')
    parser.add_argument('--epoch', default=60, type=int, help='training epoch')
    parser.add_argument('--file_name', default='training', type=str, help='file name to save weights and log file')
    opt = parser.parse_args()

    """Save Log History"""
    sys.stdout = logging.Logger(os.path.join(opt.file_name +'/'+opt.name+'/', 'log.txt'))

    """Set GPU"""
    gpu_ids = []
    str_gpu_ids = opt.gpu_ids.split(',')
    for str_id in str_gpu_ids:
        gpu_ids.append(int(str_id))
    # torch.cuda.set_device(device)
    use_gpu = torch.cuda.is_available()
    device  = torch.device('cuda' if use_gpu else 'cpu') 
    cudnn.enabled = True
    cudnn.benchmark = True

    """Person ReId Model"""
    reid_model = EvReId(class_num=opt.num_ids, num_channel=opt.num_Bin, AE_block=opt.An_model_block)

    """Optimizer"""
    optimizer = set_optimizer(reid_model)
    # reid_model = reid_model.cuda()
    reid_model = reid_model.to(device)
    
    """Ëvent-to-Video Model"""
    E2VID = load_E2VID(opt.e2vid_path, device)
    # E2VID = E2VID.cuda()
    E2VID = E2VID.to(device)
    intensity_rescaler = IntensityRescaler()
    # un_mask_filter = UnsharpMaskFilter(device=torch.device('cuda:0'))
    un_mask_filter = UnsharpMaskFilter(device=device)

    """Set ReId Loss function"""
    criterion_triplet = TripletLoss(opt.margin)
    criterion_softmax = nn.CrossEntropyLoss()

    """Save Dir"""
    dir_name = os.path.join('./' + opt.file_name, opt.name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    # save opts
    with open('%s/opts.json'%dir_name, 'w') as fp:
        json.dump(vars(opt), fp, indent=1)

    """Start Training"""
    model_ = train_model(reid_model, E2VID, optimizer, None, device, num_epochs=opt.epoch)
