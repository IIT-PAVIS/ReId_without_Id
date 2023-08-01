from math import ceil, floor
from torch.nn import ReflectionPad2d
import numpy as np
import torch
import cv2
from collections import deque
import atexit
import scipy.stats as st
import torch.nn.functional as F
from math import sqrt


def gkern(kernlen=5, nsig=1.0):
    """Returns a 2D Gaussian kernel array."""
    """https://stackoverflow.com/a/29731818"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel).float()
    

class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """

    def __init__(self):
        #self.auto_hdr = options.auto_hdr
        self.intensity_bounds = deque()
        self.auto_hdr_median_filter_size = 10
        # self.Imin = options.Imin
        # self.Imax = options.Imax

    def __call__(self, img):
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        #if self.auto_hdr:
        #with CudaTimer('Compute Imin/Imax (auto HDR)'):
        Imin = torch.min(img).item()
        Imax = torch.max(img).item()

        # ensure that the range is at least 0.1
        Imin = np.clip(Imin, 0.0, 0.45)
        Imax = np.clip(Imax, 0.55, 1.0)

        # adjust image dynamic range (i.e. its contrast)
        if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
            self.intensity_bounds.popleft()

        self.intensity_bounds.append((Imin, Imax))
        self.Imin = torch.median([rmin for rmin, rmax in self.intensity_bounds])
        self.Imax = torch.median([rmax for rmin, rmax in self.intensity_bounds])

        #with CudaTimer('Intensity rescaling'):
        img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
        img.clamp_(0.0, 255.0)
        img = img.byte()  # convert to 8-bit tensor

        return img


class UnsharpMaskFilter:
    """
    Utility class to perform unsharp mask filtering on reconstructed images.
    """

    def __init__(self, device):
        self.unsharp_mask_amount = 0.3
        self.unsharp_mask_sigma = 1.0
        self.gaussian_kernel_size = 5
        self.gaussian_kernel = gkern(self.gaussian_kernel_size,
                                     self.unsharp_mask_sigma).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, img):
        if self.unsharp_mask_amount > 0:
            #with CudaTimer('Unsharp mask'):
            blurred = F.conv2d(img, self.gaussian_kernel,
                               padding=self.gaussian_kernel_size // 2)
            img = (1 + self.unsharp_mask_amount) * img - self.unsharp_mask_amount * blurred
        return img


