import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
import time
import ImageAugme
from scipy.misc import imread
from argparse import ArgumentParser

graph,tfArgs,inputImage,previousImage=0,0,0,0

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def compress(input_path, refer_frame_interval, output_path):
    # get video name and create corresponding dictionary
    video_name = input_path.split('/')[-1][0:-4]
    tf.io.gfile.mkdir(output_path+video_name)

    # video compress
    # 1 load vedio and get information
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # parse video to frame, set I and P frame and used DVC to compress
    grouplabel = framelabel = 0
    refer_frame = current_path = 0
    crop_size = (832, 448)
    while(cap.isOpened):
        if framelabel == refer_frame_interval:  # If already compress a group, begin a new group
            framelabel = 0
            grouplabel += 1

        (flag, frame) = cap.read()  # get current frame
        if not flag:
            break

        if framelabel == 0:
            refer_frame = frame
            current_path = output_path+video_name+'/'+str(grouplabel)
            tf.io.gfile.mkdir(current_path)
            refer_frame = cv2.resize(refer_frame, crop_size, interpolation=cv2.INTER_CUBIC)
            refer_frame=ImageAugme.augment_one_img(refer_frame)
            cv2.imwrite(current_path+'/'+'Iframe.png', refer_frame,[cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            fileName_prefix = current_path+'/'+str(framelabel)+'/'
            frame = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(current_path+'/'+str(framelabel)+'.png',frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

            with tf.Session(graph=graph) as sess:
                im1 = frame
                im2 = refer_frame
                im1 = im1 / 255.0
                im2 = im2 / 255.0
                im1 = np.expand_dims(im1, axis=0)
                im2 = np.expand_dims(im2, axis=0)

                bpp_est, Res_q, Res_prior_q, motion_q, psnr_val, recon_val = sess.run(
                    tfArgs, feed_dict={
                        inputImage: im1,
                        previousImage: im2
                    })

            if not os.path.exists( fileName_prefix):
                os.mkdir( fileName_prefix)

            output = open( fileName_prefix + 'quantized_res_feature.pkl', 'wb')
            pickle.dump(Res_q, output)

            output = open(
                 fileName_prefix + 'quantized_res_prior_feature.pkl', 'wb')
            pickle.dump(Res_prior_q, output)

            output = open( fileName_prefix + 'quantized_motion_feature.pkl', 'wb')
            pickle.dump(motion_q, output)
        framelabel += 1


def main(loadmodel, input_path, refer_frame_interval, output_path):
    global graph,tfArgs,inputImage,previousImage

    graph = load_graph(loadmodel)
    prefix = 'import/build_towers/tower_0/train_net_inference_one_pass/train_net/'

    Res = graph.get_tensor_by_name(prefix + 'Residual_Feature:0')
    Res_prior = graph.get_tensor_by_name(prefix + 'Residual_Prior_Feature:0')
    motion = graph.get_tensor_by_name(prefix + 'Motion_Feature:0')
    bpp = graph.get_tensor_by_name(prefix + 'rate/Estimated_Bpp:0')
    psnr = graph.get_tensor_by_name(prefix + 'distortion/PSNR:0')
    # reconstructed frame
    reconframe = graph.get_tensor_by_name(prefix + 'ReconFrame:0')
    inputImage = graph.get_tensor_by_name('import/input_image:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')
    tfArgs=[bpp, Res, Res_prior, motion, psnr, reconframe]

    for name in os.listdir(input_path):
        compress( input_path+name, refer_frame_interval, output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-M', type=str, dest="loadmodel",
                        default='./model/L2048/frozen_model_E.pb', help="encoder model")
    parser.add_argument('-I', type=str, dest="input_path",
                        default='./sourceVideo/', help="input video path")
    parser.add_argument('-T', type=int, dest="refer_frame_interval",
                        default=30, help="refer image interval")
    parser.add_argument('-O', type=str, dest="output_path",
                        default='./video/', help="output pkl folder")

    time_start = time.time()
    args = parser.parse_args()
    main(**vars(args))
    time_end = time.time()
    print('time cost', time_end-time_start, 's')
