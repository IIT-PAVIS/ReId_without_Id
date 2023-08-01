
import sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset
#from torchvision import transforms
from PIL import Image
import pdb
import os
from torch import nn
from torch.nn import functional as F
import functools
from torch.nn import init
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def extract_bbox(event):
    """Image"""
    xi = event[:, 1].astype(int)
    yi = event[:, 2].astype(int)
    event_frame = np.zeros((480, 640, 3), np.uint8)

    """replace event coordinates with non-zero value"""
    event_frame[yi, xi, :] = [255, 255, 255]

    """convert np.array to image"""
    event_frame = Image.fromarray(event_frame)

    """extract bbox"""
    x1, y1, x2, y2 = event_frame.getbbox()

    """Bbox Padding"""
    if x1 > 4:
        x1 = x1-2
    if y1 > 4:
        y1 = y1-2
    if y2 < 476:
        y2 = y2 + 2
    if x2 < 636:
        x2 = x2 + 2
    bbox = x1, y1, x2, y2

    return bbox


def normalize_voxel(events):
    nonzero_ev = (events != 0)
    num_nonzeros = nonzero_ev.sum()

    if num_nonzeros > 0:
        """ compute mean and stddev of the **nonzero** elements of the event tensor
        we do not use PyTorch's default mean() and std() functions since it's faster
        to compute it by hand than applying those funcs to a masked array """
        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        events = mask * (events - mean) / stddev

    return events


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation from a set of events in the time domain.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """
    width = int(width)
    height = int(height)
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float64).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


class voxelDataset(Dataset):
    
    def __init__(self, event_dir="data/sample_data/", ref_rgb_dir="sample_data/rgb/", mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.event_dir = event_dir
        self.ref_rgb_dir = ref_rgb_dir
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        self.name_list = np.genfromtxt(event_dir + self.mode + '_list.txt', dtype=str, delimiter=' ', usecols=[0])
        self.label_list = np.genfromtxt(event_dir + self.mode + '_list.txt', dtype=int, delimiter=' ', usecols=[1])
        #self.rgb_name_list = np.genfromtxt(ref_rgb_dir + self.mode + '_list.txt', dtype=str, delimiter=' ', usecols=[0])

    def __getitem__(self, index):
        #ref_rgb_img = Image.open(self.ref_rgb_dir + self.name_list[index])
        events = np.loadtxt(self.event_dir + self.name_list[index], dtype=np.float64, delimiter=' ',
                            usecols=(0, 1, 2, 3))

        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1]) # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2]) # y - y_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1])+1, height=max(events[:, 2])+1)

        """Numpy to Tensor"""
        voxel_grid = torch.tensor(voxel_grid)

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(0)

        label = self.label_list[index]

        return voxel_grid, label  #, ref_rgb_img

    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':
    pass
