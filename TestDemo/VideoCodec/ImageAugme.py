import os
import cv2
from PIL import Image
import numpy as np
import skvideo.io
from data_augmentation.video_augmentation import *


def load_one_img(path):
    image_list = []
    for i in range(1):
        # img = Image.open(path).convert('RGB')

        img = Image.open(path).convert('HSV')
        image_list.append(img)
    return np.stack(image_list)


def augmentation(video):
    sometimes = lambda aug: Sometimes(0.5, aug)
    transform = Sequential([
        # RandomRotate(degrees=10),
        # sometimes(InvertColor()),
        Add(int(random.random() * 30)),
        Multiply(0.8),
        Salt(30),
        Pepper(30),
        Superpixel(0.5, 5)

    ])
    return transform(video)


def save_video(video, output_path):
    writer = skvideo.io.FFmpegWriter(output_path,
                                     outputdict={'-b': '300000000'})
    for frame in video:
        writer.writeFrame(frame)


def augment_one_img(path):
    img = load_one_img(path)
    img = augmentation(img)
    return img[0]


if __name__ == '__main__':
    dir = '/home/chuhw/CV/DVC/TestDemo/VideoCodec/image/im001.png'
    output_path = '/home/chuhw/CV/DVC/TestDemo/VideoCodec/image/im0011.png'
    res=augment_one_img(dir)
    cv2.imwrite(output_path,res)

    # video = load_one_img(dir)
    # augmen_video = augmentation(video)
    # save_video(augmen_video, output_path)