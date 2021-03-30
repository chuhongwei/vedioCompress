import cv2
import os
import pickle
import numpy as np
import tensorflow as tf

from scipy.misc import imread
from argparse import ArgumentParser

from Encoder import encoder
from Decoder import decoder

def compress(loadmodel,input_path,refer_frame_interval,output_path):
    #get video name and create corresponding dictionary
    video_name=input_path.split('/')[-1][0:-4]
    tf.gfile.MkDir(output_path+video_name)

    # video compress
    # 1 load vedio and get information
    cap = cv2.VideoCapture("./testVideo.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)#帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #parse video to frame, set I and P frame and used DVC to compress 
    grouplabel=framelabel=0
    refer_frame=current_path=0
    while(isOpened):
        if framelabel == refer_frame_interval: #If already compress a group, begin a new group 
            framelabel=0
            grouplabel+=1

        (flag,frame) = cap.read() #get current frame
        if not flag:
            break

        if framelabel == 0:
            refer_frame=frame
            current_path=output_path+video_name+'/'+str(grouplabel)
            tf.gfile.MkDir(current_path)
            cv2.imwrite(current_path+'/'+'Iframe.png',refer_frame,[cv2.IMWRITE_JPEG_QUALITY,100])
        else:
            fileName_prefix = current_path+'/'+str(framelabel)
            encoder(loadmodel, input_path, current_path+'/'+'Iframe.png', fileName_prefix):
        
        framelabel+=1



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_E.pb', help="encoder model")
    parser.add_argument('--input_video', type=str, dest="input_path", default='./video/testVideo.mp4', help="input video path")
    parser.add_argument('--Iframe_interval', type=int, dest="refer_frame_interval", default=30, help="refer image interval")
    parser.add_argument('--output_folder', type=str, dest="output_path", default='./pkl/', help="output pkl folder")

    args = parser.parse_args()
    compress(**vars(args))