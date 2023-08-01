
from __future__ import print_function, division
import argparse
import json
import os
import pdb
import sys
import scipy.io
import time
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


def test_reid(model):
    model = model.eval()
    print('-' * 10)
    print('test model now...')
    dataloaders, image_datasets = load_test_data()

    gallery_path = image_datasets['gallery'].name_list
    query_path = image_datasets['query'].name_list
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    gallery_feature, gallery_feature_embed = extract_feature(model, dataloaders['gallery'])
    query_feature, query_feature_embed = extract_feature(model, dataloaders['query'])

    #Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,
              'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),
              'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat',result)
    result = scipy.io.loadmat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat')

    query_feature = result['query_f']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('Pool5 top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

    result = {'gallery_f':gallery_feature_embed.numpy(),'gallery_label':gallery_label,
              'gallery_cam':gallery_cam,'query_f':query_feature_embed.numpy(),
              'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat',result)
    result = scipy.io.loadmat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat')

    query_feature = result['query_f']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('embed top1:%f top5:%f top10:%f mAP:%f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))


def load_test_data():

    if opt.represent == "voxel":
        image_datasets = {x: voxelDataset(mode=x) for x in ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=False, num_workers=8) for x in ['gallery', 'query']}
        return dataloaders, image_datasets


def load_network(network, path):
    pretrained_dict = torch.load(path)
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


if __name__ == '__main__':

    """Parser"""
    parser = argparse.ArgumentParser(description='Test ReId')
    parser.add_argument('--model_path', default='training/net_59.pth', type=str, help='path to pretrained Event-ReId model wieghts')
    parser.add_argument('--represent', default='voxel', type=str, help='representation of events for reid')
    parser.add_argument('--An_model_block', default=True, help='set True, if implement Event-voxel Anonymization Block')
    parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--name', default='AN_Event_ReId', type=str, help='output model name')
    parser.add_argument('--num_ids', default=22, type=int, help='number of identities')
    parser.add_argument('--num_channel', default=5, type=int, help='number of temporal bins of event-voxel')
    parser.add_argument('--file_name', default='test_results', type=str, help='log file name')
    parser.add_argument('--mat', default='', type=str, help='name for saving representation')
    opt = parser.parse_args()

    """Save Log History"""
    sys.stdout = logging.Logger(os.path.join(opt.file_name+'/'+opt.name+'/', 'log.txt'))

    """Set GPU"""
    gpu_ids = []
    str_gpu_ids = opt.gpu_ids.split(',')
    for str_id in str_gpu_ids:
        gpu_ids.append(int(str_id))
    torch.cuda.set_device(0)
    use_gpu = torch.cuda.is_available()
    cudnn.enabled = True
    cudnn.benchmark = True

    """re-id Model"""
    reid_model = EvReId(class_num=22, num_channel=opt.num_channel, AE_block=opt.An_model_block)
    reid_model = load_network(reid_model, opt.model_path)
    reid_model = reid_model.cuda()

    """Save Dir"""
    dir_name = os.path.join('./' + opt.file_name, opt.name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    # save opts
    with open('%s/opts.json'%dir_name, 'w') as fp:
        json.dump(vars(opt), fp, indent=1)

    """Start Test"""
    test_reid(reid_model)
