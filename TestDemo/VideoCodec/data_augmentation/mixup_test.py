import os
from PIL import Image
import numpy as np
import skvideo.io


def load_one_video(dir):
    image_list = []
    for i in range(1, 31):
        img = Image.open(os.path.join(dir, '{:05d}.jpg'.format(i))).convert('RGB')
        image_list.append(img)
    return np.stack(image_list)


def TemporalMixup(alpha, va, vb):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = int(lam*30)
    mixed = va
    mixed[:index, :, :, :] = vb[:index, :, :, :]
    return mixed


def SpatialMixup(alpha, va, vb):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    va_len = va.shape[0]
    mixed = va
    for i in range(va_len):
        mixed[i, :, :, :] = lam * mixed[i, :, :, :] + (1-lam) * vb[1, :, :, :]
    return mixed


def MixUp(alpha, va, vb):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * va + (1 - lam) * vb
    return mixed_x


def augmentation(va, vb):
    # mixup = MixUp(1, va, vb)
    # mixup = TemporalMixup(1, va, vb)
    mixup = SpatialMixup(1, va, vb)
    return mixup


def save_video(video, output_path):
    writer = skvideo.io.FFmpegWriter(output_path,
                                     outputdict={'-b': '300000000'})
    for frame in video:
        writer.writeFrame(frame)


if __name__ == '__main__':
    a_dir = '../samples/resize_841_1280/'
    b_dir = '../samples/resize_40346_1280/'
    # output_path = '../output/temporal_mixup_augmentation.mp4'
    output_path = '../output/spatial_mixup_augmentation.mp4'
    a_video = load_one_video(a_dir)
    b_video = load_one_video(b_dir)
    augmen_video = augmentation(a_video, b_video)
    print(augmen_video.shape)
    save_video(augmen_video, output_path)
