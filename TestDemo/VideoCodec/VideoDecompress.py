import cv2
import os
import pickle
import numpy as np
import tensorflow as tf

from scipy.misc import imread
from argparse import ArgumentParser

from Encoder import encoder
from Decoder import decoder


def decompress(loadmodel,input_path,refer_frame_interval,output_path):
    #get video name and create corresponding dictionary
    video_name=input_path.split('/')[-2]

    # video decompress
    # create the video and wait to write frame
    crop_size = (832, 448)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
    videowrite = cv2.VideoWriter(output_path,fourcc,24,crop_size)

    grouplabel=framelabel=0
    while tf.gfile.Exists(input_path+str(grouplabel)):
        current_path=input_path+str(grouplabel)+'/'
        refer_frame = cv2.imread(current_path+'Iframe.png')
        framelabel=1
        while tf.gfile.Exists(current_path+str(framelabel)):
            decoder(loadmodel, current_path+'Iframe.png', current_path+str(framelabel)+'/',current_path+str(framelabel)+'/res.png')
            img = cv2.imread(current_path+str(framelabel)+'/res.png')
            videowrite.write(img)
            framelabel+=1
        grouplabel+=1
        if grouplabel==7:
            break



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--DecoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_D.pb', help="decoder model")
    parser.add_argument('--input_path', type=str, dest="input_path", default='./video/testVideo/', help="compressed video path")
    parser.add_argument('--Iframe_interval', type=int, dest="refer_frame_interval", default=30, help="refer image interval")
    parser.add_argument('--output_path', type=str, dest="output_path", default='/video/testVideo/', help="output video path")

    args = parser.parse_args()
    decompress(**vars(args))