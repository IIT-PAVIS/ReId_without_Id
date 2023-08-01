
from PIL import Image
import numpy as np
import cv2
import pdb
import os
import time
import signal
import argparse
import json
import shutil


def split_data(path_, set_):

    path_ = path_ + "/"
    file_array = [file for file in os.listdir(path_) if file.endswith('.txt')]
    file_array = sorted(file_array)
    
    if set_ == "train":
       split_save_dir = "train/"
       step = 1
    else:   
       split_save_dir = "test/"
       step = 5

    if not os.path.exists(split_save_dir):
        os.makedirs(split_save_dir)

    for k in range(0, len(file_array), step):
        copy_path = path_ + file_array[k]
        save_path = split_save_dir + file_array[k]
        shutil.copyfile(copy_path, save_path)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='out_dir/Event_ReId/')
    params = parser.parse_args()

    for i in range(33):
            i_d = "{0:03}".format(i+1)
            path = params.data_dir + str(i_d) + '/'
            sub_path = os.listdir(path)

            for j in range(4):
                complete_path = os.path.join(path, sub_path[j])
                id_ = i+1
                if id_ not in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]:
                   set_ = "train"
                else:
                   set_ = "test"
                split_data(complete_path, set_)
   

if __name__ == "__main__":
    main()
