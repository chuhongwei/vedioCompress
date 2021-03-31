from PIL import Image
from numpy import average, dot, linalg
import cv2

def diatance_ratio(v1,v2):
    distance=0
    avg=0
    for i in range(0,3):
        avg+=(0+v1[i]+v2[i])/2
        distance+=abs(0+v1[i]-v2[i])
    
    return distance/(avg+1)



def cal_distance(im1,im2):
    distance=0
    M,N,K=im1.shape
    for i in range(0,M):
        for j in range(0,N):
            distance+=diatance_ratio(im1[i][j],im2[i][j])
    
    return distance/(len(im1)*len(im2))
            


cap1 = cv2.VideoCapture('./testVideo.mp4')
cap2 = cv2.VideoCapture('./video/testVideo/testVideo.mp4')
(flag1,frame1) = cap1.read()
frame1 = cv2.resize(frame1, (832,448), interpolation = cv2.INTER_CUBIC)
(flag2,frame2) = cap2.read()

i=0
while flag1 and flag2:

    frame1 = cv2.resize(frame1, (832,448), interpolation = cv2.INTER_CUBIC)
    cosin = cal_distance(frame1, frame2)
    print('第',i,'帧的差别占比',cosin,'%')
    (flag1,frame1) = cap1.read()
    (flag2,frame2) = cap2.read()
    i+=1
