import torch
import numpy as np
from torch.autograd import Variable
import scipy.io
import pdb


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_feature(model,dataloaders):
    pool5_features = torch.FloatTensor()
    embed_features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            _, embed_feature, pool5_feature,_ = model(input_img)
            if(i==0):
                ff_pool5 = torch.FloatTensor(n,pool5_feature.size(1)).zero_()
                ff_embed = torch.FloatTensor(n,embed_feature.size(1)).zero_()
            f_pool5 = pool5_feature.data.cpu()
            ff_pool5 = ff_pool5 + f_pool5
            f_embed = embed_feature.data.cpu()
            ff_embed = ff_embed + f_embed
        fnorm_pool5 = torch.norm(ff_pool5, p=2, dim=1, keepdim=True)
        fnorm_embed = torch.norm(ff_embed, p=2, dim=1, keepdim=True)
        ff_pool5 = ff_pool5.div(fnorm_pool5.expand_as(ff_pool5))
        ff_embed = ff_embed.div(fnorm_embed.expand_as(ff_embed))
        pool5_features = torch.cat((pool5_features,ff_pool5), 0)
        embed_features = torch.cat((embed_features,ff_embed), 0)
    return pool5_features, embed_features


def get_id(img_path):
    camera_id = []
    labels = []
    for path in img_path:
        filename = path.split('/')[-1] #001_c1_035.txt
        label = filename.split('_')[0]
        
        camera = filename.split('c')[1]
        
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0])) 
        #camera_id.append(int(camera[1]))
 
    return camera_id, labels


def evaluate_test(qf,ql,qc,gf,gl,gc):
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    #if not good_index.size==0:
 
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = []

    for i in range(len(gl)):
        if gl[i] == ql:
            query_index.append(i)

    camera_index = []

    for i in range(len(gc)):
        if gc[i] == qc:
            camera_index.append(i)


    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    #if not good_index.size==0:
   
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
